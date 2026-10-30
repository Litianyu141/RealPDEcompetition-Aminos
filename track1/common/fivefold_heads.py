"""Strict, opt-in data contract shared by vgain and learned-band producers.

Head-plan paths are repository-relative (absolute paths also work). Checkpoints
name hash-bound complete provenance sidecars using ``common.fivefold``'s schema.
The plan's fit_files are real basenames. An inner manifest can contain more rows;
only explicit fit_files are consumed. No cache without this plan stamp is reused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    from .fivefold import load_protocol, _validate_provenance
except ImportError:
    from fivefold import load_protocol, _validate_provenance

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / 'data/RealPDE_competition_dataset/gdrive_raw/extracted/train_real/train_real'


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _path(value):
    p = Path(value)
    return p if p.is_absolute() else REPO / p


def select_fit_files(files, allowlist):
    names = [Path(f).name for f in files]
    if len(names) != len(set(names)):
        raise ValueError('duplicate fitting file')
    allowed = set(allowlist)
    return [f for f in files if Path(f).name in allowed]


def strict_selftest_inputs(cache_inputs):
    """Never fall back to the legacy val stream, even for an empty cache."""
    n = min(8, len(cache_inputs))
    if n == 0:
        raise ValueError('empty strict selftest cache')
    return cache_inputs[:n]


class HeadPlan:
    def __new__(cls, path):
        # Final fitting has its own contract; never synthesize a sixth OOF fold.
        if cls is HeadPlan:
            data = json.loads(_path(path).read_text())
            if data.get('purpose') == 'final_fit':
                try:
                    from .finalfit_heads import FinalFitHeadPlan
                except ImportError:
                    from finalfit_heads import FinalFitHeadPlan
                return FinalFitHeadPlan(path)
        return super().__new__(cls)

    def __init__(self, path):
        self.path = _path(path)
        self.data = json.loads(self.path.read_text())
        self.protocol = load_protocol(_path(self.data['protocol_manifest']))
        matches = [f for f in self.protocol['folds'] if f['id'] == self.data['outer_fold']]
        if len(matches) != 1:
            raise ValueError('unknown outer fold')
        self.outer = matches[0]
        self.manifest_path = _path(self.data['inner_manifest'])
        self.manifest = json.loads(self.manifest_path.read_text())
        self.fitting_files = self.data['fit_files']
        if (not self.fitting_files or len(set(self.fitting_files)) != len(self.fitting_files)
                or any(Path(x).name != x for x in self.fitting_files)):
            raise ValueError('fit_files must be nonempty unique basenames')
        if not set(self.fitting_files) <= set(self.outer['train_files']):
            raise ValueError('fitting files contain outer holdout contamination')
        self.folds = []
        seen = set()
        ids = set()
        for fd in self.manifest['folds']:
            k = int(fd['fold'])
            if k in ids:
                raise ValueError('duplicate inner fold id')
            ids.add(k)
            files = select_fit_files(fd['val_files'], self.fitting_files)
            if not files:
                continue
            names = {Path(f).name for f in files}
            if seen & names:
                raise ValueError('duplicate inner validation file')
            seen.update(names)
            # File identities select released data; a manifest cannot redirect a
            # familiar basename to an arbitrary external dataset.
            released = [str(REAL / Path(f).name) for f in files]
            for f in released:
                p = Path(f)
                if p.exists() and not p.resolve().is_relative_to(REPO.resolve()):
                    raise ValueError('strict fitting source resolves outside repository')
            self.folds.append(dict(fd, val_files=released))
        if seen != set(self.fitting_files):
            raise ValueError('inner manifest does not cover every fitting file')

    def checkpoint(self, arm, fold):
        return _path(self.data['checkpoints'][arm][str(fold)]['path'])

    def _verify(self, arm, fd):
        item = self.data['checkpoints'][arm][str(fd['fold'])]
        cp = self.checkpoint(arm, fd['fold'])
        sidecar = _path(item['provenance'])
        if sha256_file(cp) != item['sha256'] or sha256_file(sidecar) != item['provenance_sha256']:
            raise ValueError('producer checkpoint/provenance hash mismatch')
        prov = json.loads(sidecar.read_text())
        bundle = {'protocol_id': self.protocol['protocol_id'],
                  'method': prov.get('method'), 'track': prov.get('track')}
        _validate_provenance(prov, sidecar.parent, self.outer, bundle)
        artifacts = {a['id']: a for a in prov['artifacts']}
        components = {c['id']: c for c in prov['components']}
        root_outputs = [artifacts[a] for c in prov['root_components']
                        for a in components[c]['output_artifact_ids']]
        def artifact_path(a):
            p = Path(a['path'])
            return p if p.is_absolute() else sidecar.parent / p
        if not any(artifact_path(a).resolve() == cp.resolve() and a['sha256'] == item['sha256']
                   for a in root_outputs):
            raise ValueError('checkpoint is not a provenance root output')
        heldout = {Path(f).name for f in fd['val_files']}
        axis = self.manifest.get('group_by', 're')
        axis_index = {'re': 0, 'aoa': 1}[axis]
        groups = set(fd.get('val_groups') or fd.get('val_re') or [])
        for component in prov['components']:
            for stage in ('train_files', 'fit_files', 'selection_files'):
                for ref in component[stage]:
                    name = ref['name']
                    group = int(name[:-3].split('_')[axis_index])
                    if (ref['domain'] == 'real' and name in heldout) or group in groups:
                        raise ValueError('inner OOF contamination in producer or teacher')
        return item

    def stamp(self, arms):
        verified = {arm: {str(fd['fold']): self._verify(arm, fd) for fd in self.folds}
                    for arm in sorted(set(arms))}
        payload = {'plan': sha256_file(self.path),
                   'protocol': sha256_file(_path(self.data['protocol_manifest'])),
                   'manifest': sha256_file(self.manifest_path), 'producers': verified}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def sband_stamp(plan, fleet_a, fleet_b):
    """Bind the two blend partitions, not only their flattened set of members."""
    if not fleet_a or not fleet_b:
        raise ValueError('both sband fleets must be nonempty')
    value = {'producers': plan.stamp(list(fleet_a)+list(fleet_b)),
             'fleet_a': sorted(fleet_a), 'fleet_b': sorted(fleet_b)}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_cache_metadata(cache_dir, plan, arms, fleet_a=None, fleet_b=None):
    meta = json.loads((Path(cache_dir) / 'meta.json').read_text())
    expected = (sband_stamp(plan, fleet_a, fleet_b) if fleet_a is not None else plan.stamp(arms))
    if meta.get('strict_plan_stamp') != expected:
        raise ValueError('strict cache plan/producer stamp mismatch')
    actual = [name for group in meta['files'] for name in group]
    if len(actual) != len(set(actual)) or set(actual) != set(plan.fitting_files):
        raise ValueError('strict cache fitting files mismatch')
    arrays = meta.get('array_sha256', {})
    required = {'Xp.npy', 'Rkf.npy', 'Rks.npy', 'SC.npy', 'grp.npy', 'afac.npy'}
    if set(arrays) != required:
        raise ValueError('strict cache array hashes missing')
    for name, digest in arrays.items():
        if sha256_file(Path(cache_dir) / name) != digest:
            raise ValueError('strict cache array hash mismatch')
    return meta


def neutral_t1_base():
    """Data-independent fallback, inert when vgain and learned sband are active."""
    import numpy as np
    return {'gamma': np.ones((32, 64, 3), np.float32),
            'band_lo': np.full((20, 3), -0.15, np.float32),
            'band_hi': np.full((20, 3), 0.15, np.float32),
            'k_cell': np.ones((20, 3), np.float32),
            'act_const': np.ones(3, np.float32), 'p': np.float32(0),
            'lam': np.float32(1), 'self_norm': np.bool_(True),
            'act_kind': np.str_('std')}
