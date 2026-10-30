#!/usr/bin/env python3
"""Fixed Re3 + AoA2 protocol and auditable out-of-fold evidence validation.

The scorer consumes the same per-window sufficient statistics as
``local_benchmark/splitsel/score_card.py``.  It deliberately does not consume
that scorer's ``all81`` summary: every row is checked against the outer
holdout named by the immutable protocol before it is aggregated.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "local_benchmark" / "splits_re3_aoa2_v1.json"
DEFAULT_DATA_SNAPSHOT = HERE / "local_benchmark" / "re3_aoa2_data_snapshot_v1.json"
# Set only from the measured data snapshot; changing snapshot+manifest alone is refused.
DATA_SNAPSHOT_SHA256 = "5d48473a8055a4b198ec4b4406e9fc6ff01a83235261ff5d7d121c1b17f46563"

RE_GROUPS = {
    "re0": [3750, 8850, 11400, 17775, 20325, 25425],
    "re1": [5025, 10125, 12675, 19050, 21600, 26700],
    "re2": [6300, 7575, 13950, 16500, 22875, 24150],
}
AOA_GROUPS = {"aoa0": [0, 10, 20], "aoa1": [5, 15]}
WINDOW_COLUMNS = ("file", "t0", "rel", "tke", "mvpe", "S", "n", "nin")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _basename(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError("%s must contain h5 basenames, got %r" % (label, value))
    parts = value[:-3].split("_") if value.endswith(".h5") else []
    if len(parts) != 2:
        raise ValueError("%s has invalid h5 basename %r" % (label, value))
    try:
        int(parts[0]); int(parts[1])
    except ValueError as exc:
        raise ValueError("%s has invalid h5 basename %r" % (label, value)) from exc
    return value


def _coords(name: str) -> Tuple[int, int]:
    re_value, aoa = name[:-3].split("_")
    return int(re_value), int(aoa)


def _unique_strings(values: Any, label: str, *, basenames: bool = False) -> List[str]:
    if not isinstance(values, list):
        raise ValueError("%s must be a list" % label)
    result = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("%s must contain strings" % label)
        result.append(_basename(value, label) if basenames else value)
    if len(result) != len(set(result)):
        raise ValueError("%s contains duplicates" % label)
    return result


def load_protocol(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load and validate the fixed protocol manifest."""
    source = Path(path) if path is not None else DEFAULT_PROTOCOL
    with source.open(encoding="utf-8") as stream:
        protocol = json.load(stream)
    validate_protocol(protocol)
    return protocol


def load_data_snapshot(protocol: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Read the measured, byte-pinned source universe and evaluation grid."""
    if DATA_SNAPSHOT_SHA256 is None:
        raise ValueError("measured data snapshot has not been published")
    if protocol is None:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text())
    binding = protocol.get("data_snapshot")
    if binding != {"path": DEFAULT_DATA_SNAPSHOT.name, "sha256": DATA_SNAPSHOT_SHA256}:
        raise ValueError("protocol data snapshot binding differs from the frozen digest")
    if _sha_file(DEFAULT_DATA_SNAPSHOT) != DATA_SNAPSHOT_SHA256:
        raise ValueError("data snapshot sha256 differs from the frozen digest")
    snapshot = json.loads(DEFAULT_DATA_SNAPSHOT.read_text())
    if (snapshot.get("schema_version") != 1 or snapshot.get("protocol_id") != "re3_aoa2_v1"
            or snapshot.get("t_in") != 20 or snapshot.get("t_out") != 20
            or snapshot.get("stride_eval") != 20):
        raise ValueError("unsupported data snapshot schema or evaluation window parameters")
    for domain, count in (("real", 81), ("sim", 100)):
        entries = snapshot.get(domain)
        if not isinstance(entries, dict) or len(entries) != count:
            raise ValueError("data snapshot has incorrect released %s universe" % domain)
        for name, item in entries.items():
            _basename(name, "data snapshot file")
            shape = item.get("shape")
            if (not isinstance(shape, list) or len(shape) != 3
                    or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)
                    or shape[0] < 40 or shape[1:] != [64, 128]):
                raise ValueError("data snapshot has invalid u/v shape for " + name)
            if item.get("eval_t0") != list(range(0, shape[0] - 40 + 1, 20)):
                raise ValueError("data snapshot evaluation window grid disagrees with shape for " + name)
            digest = item.get("sha256")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                    or not isinstance(item.get("size"), int) or item["size"] <= 0):
                raise ValueError("data snapshot file sha256/size invalid for " + name)
    return snapshot


def expected_eval_windows(fold: Mapping[str, Any],
                          protocol: Optional[Mapping[str, Any]] = None) -> List[Tuple[str, int]]:
    """Exact released (file,t0) population, t_in=20/t_out=20/stride=20."""
    snapshot = load_data_snapshot(protocol)
    return [(name, t0) for name in fold["val_files"]
            for t0 in snapshot["real"][name]["eval_t0"]]


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject malformed, randomized, incomplete, or leaking protocol manifests."""
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be an object")
    if protocol.get("protocol_id") != "re3_aoa2_v1":
        raise ValueError("protocol_id must be re3_aoa2_v1")
    real = _unique_strings(protocol.get("real_files"), "real_files", basenames=True)
    sim = _unique_strings(protocol.get("sim_files"), "sim_files", basenames=True)
    if len(real) != 81:
        raise ValueError("protocol must contain exactly 81 legal real files")
    if len(sim) != 100:
        raise ValueError("protocol must contain exactly 100 legal sim files")
    if "7575_0.h5" in real:
        raise ValueError("real 7575_0.h5 is banned")
    if "7575_0.h5" not in sim:
        raise ValueError("sim 7575_0.h5 is legal and must be present")
    snapshot = load_data_snapshot(protocol)
    if set(real) != set(snapshot["real"]):
        raise ValueError("protocol real_files differ from the exact released real universe")
    if set(sim) != set(snapshot["sim"]):
        raise ValueError("protocol sim_files differ from the exact released sim universe")
    if len({_coords(name) for name in real}) != 81:
        raise ValueError("real_files coordinates are not unique")
    expected_sim = {"%d_%d.h5" % (re_value, aoa)
                    for re_value in range(3750, 27975 + 1, 1275)
                    for aoa in (0, 5, 10, 15, 20)}
    if set(sim) != expected_sim:
        raise ValueError("sim_files must be the exact released 20x5 grid of 100 files")

    folds = protocol.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("protocol must contain exactly five folds")
    expected_ids = list(RE_GROUPS) + list(AOA_GROUPS)
    if [fold.get("id") for fold in folds if isinstance(fold, Mapping)] != expected_ids:
        raise ValueError("fold ids/order must be %s" % expected_ids)
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise ValueError("each fold must be an object")
        fold_id = fold["id"]
        axis = fold.get("axis")
        expected_axis = "re" if fold_id in RE_GROUPS else "aoa"
        groups = RE_GROUPS.get(fold_id, AOA_GROUPS.get(fold_id))
        if axis != expected_axis or fold.get("val_groups") != groups:
            raise ValueError("%s has incorrect axis/val_groups" % fold_id)
        val = _unique_strings(fold.get("val_files"), "%s.val_files" % fold_id,
                              basenames=True)
        train = _unique_strings(fold.get("train_files"), "%s.train_files" % fold_id,
                                basenames=True)
        sim_train = _unique_strings(fold.get("sim_train_files"),
                                    "%s.sim_train_files" % fold_id, basenames=True)
        if axis == "re":
            expected_val = {name for name in real if _coords(name)[0] in groups}
            expected_sim_train = {name for name in sim if _coords(name)[0] not in groups}
        else:
            expected_val = {name for name in real if _coords(name)[1] in groups}
            expected_sim_train = {name for name in sim if _coords(name)[1] not in groups}
        if set(val) != expected_val:
            raise ValueError("%s val_files do not match its fixed groups" % fold_id)
        if set(train) != set(real) - expected_val:
            raise ValueError("%s train_files are not the exact real complement" % fold_id)
        if set(sim_train) != expected_sim_train:
            raise ValueError("%s sim_train_files do not exclude the matching sim groups" % fold_id)

    for axis in ("re", "aoa"):
        held_out = [name for fold in folds if fold["axis"] == axis
                    for name in fold["val_files"]]
        if len(held_out) != 81 or len(set(held_out)) != 81 or set(held_out) != set(real):
            raise ValueError("%s folds must cover every real file exactly once" % axis)


def protocol_hash(protocol: Optional[Mapping[str, Any]] = None) -> str:
    """Return the canonical SHA-256 identity of a validated protocol."""
    value = load_protocol() if protocol is None else protocol
    validate_protocol(value)
    return _sha_bytes(_canonical(value))


def _load_bound_json(value: Any, expected_hash: Any, base: Path, label: str
                     ) -> Tuple[Dict[str, Any], str, Path]:
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("%s_sha256 is required" % label)
    if isinstance(value, Mapping):
        raw = _canonical(value)
        resolved = base
        data = dict(value)
    elif isinstance(value, (str, Path)):
        path = Path(value)
        resolved = path if path.is_absolute() else base / path
        if not resolved.is_file():
            raise ValueError("%s does not exist: %s" % (label, resolved))
        raw = resolved.read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("%s is not valid JSON" % label) from exc
    else:
        raise ValueError("%s must be a path or object" % label)
    actual = _sha_bytes(raw)
    if actual != expected_hash:
        raise ValueError("%s sha256 mismatch: expected %s, got %s" %
                         (label, expected_hash, actual))
    if not isinstance(data, dict):
        raise ValueError("%s JSON must be an object" % label)
    return data, actual, resolved


def _verify_artifacts(provenance: Mapping[str, Any], base: Path, fold_id: str
                      ) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, str]]:
    values = provenance.get("artifacts")
    if not isinstance(values, list) or not values:
        raise ValueError("%s provenance artifacts must be a non-empty list" % fold_id)
    artifacts: Dict[str, Mapping[str, Any]] = {}
    hashes = {}
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError("%s artifact must be an object" % fold_id)
        artifact_id = item.get("id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in artifacts:
            raise ValueError("%s artifact ids must be unique non-empty strings" % fold_id)
        if item.get("role") not in ("input_manifest", "fitted_component", "selection_evidence"):
            raise ValueError("%s artifact %s has invalid role" % (fold_id, artifact_id))
        if not isinstance(item.get("path"), str) or not item["path"]:
            raise ValueError("%s artifact %s path is required" % (fold_id, artifact_id))
        path = Path(item["path"])
        resolved = path if path.is_absolute() else base / path
        if not resolved.is_file():
            raise ValueError("%s artifact %s does not exist" % (fold_id, artifact_id))
        if not isinstance(item.get("size"), int) or item["size"] != resolved.stat().st_size:
            raise ValueError("%s artifact %s size mismatch" % (fold_id, artifact_id))
        actual = _sha_file(resolved)
        if item.get("sha256") != actual:
            raise ValueError("%s artifact %s sha256 mismatch" % (fold_id, artifact_id))
        artifacts[artifact_id] = dict(item, _resolved_path=resolved)
        hashes[artifact_id] = actual
    return artifacts, hashes


def _source_ref(value: Any, label: str) -> Tuple[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"domain", "name"}:
        raise ValueError("%s entries must be {domain,name} objects" % label)
    domain = value.get("domain")
    if domain not in ("real", "sim"):
        raise ValueError("%s domain must be real or sim" % label)
    return domain, _basename(value.get("name"), label)


def _validate_input_manifest(data: Mapping[str, Any], fold: Mapping[str, Any], label: str
                             ) -> Tuple[set, int]:
    if data.get("schema_version") != 1:
        raise ValueError("%s schema_version must be 1" % label)
    files_raw = data.get("files")
    windows_raw = data.get("windows")
    if not isinstance(files_raw, list) or not isinstance(windows_raw, list):
        raise ValueError("%s files/windows must be lists" % label)
    files = [_source_ref(value, "%s.files" % label) for value in files_raw]
    if len(files) != len(set(files)):
        raise ValueError("%s contains duplicate files" % label)
    windows = []
    for value in windows_raw:
        if not isinstance(value, Mapping) or set(value) != {"domain", "file", "t0"}:
            raise ValueError("%s window entries must be {domain,file,t0}" % label)
        source = _source_ref({"domain": value["domain"], "name": value["file"]},
                             "%s.windows" % label)
        t0 = value["t0"]
        if isinstance(t0, bool) or not isinstance(t0, int) or t0 < 0:
            raise ValueError("%s window t0 must be a non-negative integer" % label)
        windows.append(source + (t0,))
    if len(windows) != len(set(windows)):
        raise ValueError("%s contains duplicate windows" % label)
    if data.get("n_files") != len(files) or data.get("n_windows") != len(windows):
        raise ValueError("%s file/window counts do not match its evidence" % label)
    represented = {(domain, name) for domain, name, _ in windows}
    if represented != set(files):
        raise ValueError("%s windows do not exactly cover its declared input files" % label)
    allowed = ({("real", name) for name in fold["train_files"]} |
               {("sim", name) for name in fold["sim_train_files"]})
    holdout = {("real", name) for name in fold["val_files"]}
    if set(files) & holdout:
        raise ValueError("%s contains outer holdout contamination" % label)
    if not set(files) <= allowed:
        raise ValueError("%s contains files outside the fold training pools" % label)
    return set(files), len(windows)


def _validate_provenance(provenance: Mapping[str, Any], provenance_base: Path,
                         fold: Mapping[str, Any], bundle: Mapping[str, Any]
                         ) -> Tuple[Dict[str, str], Dict[str, Any]]:
    fold_id = fold["id"]
    expected = {"protocol_id": bundle["protocol_id"], "method": bundle["method"],
                "track": bundle["track"], "fold_id": fold_id}
    if provenance.get("schema_version") != 1:
        raise ValueError("%s provenance schema_version must be 1" % fold_id)
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError("%s provenance %s mismatch" % (fold_id, key))
    artifacts, artifact_hashes = _verify_artifacts(provenance, provenance_base, fold_id)
    components_raw = provenance.get("components")
    if not isinstance(components_raw, list) or not components_raw:
        raise ValueError("%s provenance components must be non-empty" % fold_id)
    components = {}
    used_artifacts = set()
    component_sources = set()
    for component in components_raw:
        if not isinstance(component, Mapping):
            raise ValueError("%s component must be an object" % fold_id)
        component_id = component.get("id")
        if not isinstance(component_id, str) or not component_id or component_id in components:
            raise ValueError("%s component ids must be unique" % fold_id)
        if not isinstance(component.get("kind"), str) or not component["kind"]:
            raise ValueError("%s component %s kind is required" % (fold_id, component_id))
        parents = _unique_strings(component.get("parents"), "%s.%s.parents" %
                                  (fold_id, component_id))
        input_id = component.get("input_manifest_id")
        output_ids = _unique_strings(component.get("output_artifact_ids"),
                                     "%s.%s.output_artifact_ids" % (fold_id, component_id))
        if not output_ids:
            raise ValueError("%s component %s must bind an output artifact" %
                             (fold_id, component_id))
        if input_id not in artifacts or artifacts[input_id].get("role") != "input_manifest":
            raise ValueError("%s component %s input_manifest_id is invalid" %
                             (fold_id, component_id))
        if any(output_id not in artifacts or artifacts[output_id].get("role") == "input_manifest"
               for output_id in output_ids):
            raise ValueError("%s component %s output artifacts are invalid" %
                             (fold_id, component_id))
        manifest_path = artifacts[input_id]["_resolved_path"]
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("%s component %s input manifest is invalid JSON" %
                             (fold_id, component_id)) from exc
        manifest_sources, _ = _validate_input_manifest(
            manifest, fold, "%s component %s input manifest" % (fold_id, component_id))
        declared_sources = set()
        for stage in ("train_files", "fit_files", "selection_files"):
            values = component.get(stage)
            if not isinstance(values, list):
                raise ValueError("%s component %s %s must be a list" %
                                 (fold_id, component_id, stage))
            refs = [_source_ref(value, "%s.%s.%s" % (fold_id, component_id, stage))
                    for value in values]
            if len(refs) != len(set(refs)):
                raise ValueError("%s component %s %s contains duplicates" %
                                 (fold_id, component_id, stage))
            declared_sources.update(refs)
        holdout = {("real", name) for name in fold["val_files"]}
        if declared_sources & holdout:
            raise ValueError("%s component %s contains outer holdout contamination" %
                             (fold_id, component_id))
        if declared_sources != manifest_sources:
            raise ValueError("%s component %s staged files do not exactly match its hashed input manifest" %
                             (fold_id, component_id))
        component_sources.update(declared_sources)
        used_artifacts.add(input_id); used_artifacts.update(output_ids)
        components[component_id] = dict(component, parents=parents,
                                        _sources=declared_sources)

    roots = _unique_strings(provenance.get("root_components"),
                            "%s.root_components" % fold_id)
    if not roots or any(root not in components for root in roots):
        raise ValueError("%s root_components are invalid" % fold_id)
    visiting, reachable = set(), set()

    def visit(component_id: str) -> None:
        if component_id in visiting:
            raise ValueError("%s component parent graph contains a cycle" % fold_id)
        if component_id in reachable:
            return
        visiting.add(component_id)
        for parent in components[component_id]["parents"]:
            if parent not in components:
                raise ValueError("%s component %s has unknown parent %s" %
                                 (fold_id, component_id, parent))
            visit(parent)
        visiting.remove(component_id)
        reachable.add(component_id)

    for root in roots:
        visit(root)
    if reachable != set(components):
        raise ValueError("%s every fitted component must be reachable from a recipe root" % fold_id)

    def ancestors(component_id: str) -> set:
        result = set()
        pending = list(components[component_id]["parents"])
        while pending:
            parent = pending.pop()
            if parent not in result:
                result.add(parent)
                pending.extend(components[parent]["parents"])
        return result

    def transitive_sources(component_id: str) -> set:
        ids = ancestors(component_id) | {component_id}
        return set().union(*(components[item]["_sources"] for item in ids))

    # A fitted head/calibrator may consume OOF predictions.  Merely saying that
    # its fit pool was legal is insufficient: bind each subset to the ancestor
    # that produced it, then prove the subset was held out from that producer's
    # complete dependency closure.
    for component_id, component in components.items():
        fit_files = {_source_ref(value, "%s.%s.fit_files" % (fold_id, component_id))
                     for value in component["fit_files"]}
        evidence = component.get("fit_evidence", [])
        fit_policy = component.get("fit_policy")
        if fit_files and not evidence:
            raise ValueError("%s component %s fit_files require producer fit_evidence" %
                             (fold_id, component_id))
        if fit_files and fit_policy not in ("inner_oof", "training_predictions"):
            raise ValueError("%s component %s must label fit_policy as inner_oof or training_predictions" %
                             (fold_id, component_id))
        if not fit_files and fit_policy is not None:
            raise ValueError("%s component %s fit_policy requires non-empty fit_files" %
                             (fold_id, component_id))
        if not isinstance(evidence, list):
            raise ValueError("%s component %s fit_evidence must be a list" %
                             (fold_id, component_id))
        assigned = set()
        valid_producers = ancestors(component_id)
        for item in evidence:
            if not isinstance(item, Mapping) or set(item) != {
                    "producer_component_id", "evidence_artifact_id", "files"}:
                raise ValueError("%s component %s fit_evidence entries are invalid" %
                                 (fold_id, component_id))
            producer = item["producer_component_id"]
            artifact_id = item["evidence_artifact_id"]
            if producer not in valid_producers:
                raise ValueError("%s component %s fit producer must be an ancestor" %
                                 (fold_id, component_id))
            if (artifact_id not in artifacts or
                    artifacts[artifact_id].get("role") != "selection_evidence"):
                raise ValueError("%s component %s fit evidence artifact is invalid" %
                                 (fold_id, component_id))
            refs = [_source_ref(value, "%s.%s.fit_evidence.files" %
                                (fold_id, component_id)) for value in item["files"]]
            if not refs or len(refs) != len(set(refs)) or assigned & set(refs):
                raise ValueError("%s component %s fit evidence files must be non-empty and unique" %
                                 (fold_id, component_id))
            producer_sources = transitive_sources(producer)
            if fit_policy == "inner_oof" and set(refs) & producer_sources:
                raise ValueError("%s component %s inner holdout leaked into producer %s" %
                                 (fold_id, component_id, producer))
            if fit_policy == "training_predictions" and not set(refs) <= producer_sources:
                raise ValueError("%s component %s training_predictions were not producer inputs" %
                                 (fold_id, component_id))
            assigned.update(refs)
            used_artifacts.add(artifact_id)
        if assigned != fit_files:
            raise ValueError("%s component %s fit_evidence must exactly cover fit_files" %
                             (fold_id, component_id))
    if used_artifacts != set(artifacts):
        raise ValueError("%s provenance has unbound artifacts" % fold_id)
    recipe = {
        "fold_id": fold_id,
        "roots": roots,
        "components": [{key: component[key] for key in
                        ("id", "kind", "parents", "input_manifest_id", "output_artifact_ids",
                         "train_files", "fit_files", "selection_files", "fit_policy",
                         "fit_evidence") if key in component}
                       for component in components_raw],
        "artifact_hashes": artifact_hashes,
    }
    return artifact_hashes, recipe


def _window_rows(data: Mapping[str, Any], fold: Mapping[str, Any],
                 protocol: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    fold_id = fold["id"]
    if data.get("status") != "OK":
        raise ValueError("%s windows status must be OK" % fold_id)
    columns = data.get("columns")
    if not isinstance(columns, list) or any(name not in columns for name in WINDOW_COLUMNS):
        raise ValueError("%s windows columns must include %s" % (fold_id, WINDOW_COLUMNS))
    if len(columns) != len(set(columns)):
        raise ValueError("%s windows columns contain duplicates" % fold_id)
    rows_raw = data.get("rows")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ValueError("%s windows rows must be non-empty" % fold_id)
    rows = []
    seen = set()
    for index, raw in enumerate(rows_raw):
        if not isinstance(raw, list) or len(raw) != len(columns):
            raise ValueError("%s windows row %d has wrong width" % (fold_id, index))
        row = dict(zip(columns, raw))
        name = _basename(row["file"], "%s windows.file" % fold_id)
        t0 = row["t0"]
        if isinstance(t0, bool) or not isinstance(t0, int) or t0 < 0:
            raise ValueError("%s windows t0 must be a non-negative integer" % fold_id)
        key = (name, t0)
        if key in seen:
            raise ValueError("%s windows contain duplicate (file,t0) rows" % fold_id)
        seen.add(key)
        for key_name in ("rel", "tke", "mvpe", "S", "n", "nin"):
            value = row[key_name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("%s windows %s must be finite numeric" % (fold_id, key_name))
        if min(row["rel"], row["tke"], row["mvpe"], row["S"]) < 0:
            raise ValueError("%s windows sufficient statistics must be non-negative" % fold_id)
        if (not isinstance(row["n"], int) or not isinstance(row["nin"], int) or
                row["n"] <= 0 or not 0 <= row["nin"] <= row["n"] or row["S"] > row["n"]):
            raise ValueError("%s windows n/nin/S are inconsistent" % fold_id)
        rows.append(row)
    files = {row["file"] for row in rows}
    if files != set(fold["val_files"]):
        raise ValueError("%s windows do not exactly cover held-out files" % fold_id)
    expected = set(expected_eval_windows(fold, protocol))
    if seen != expected:
        missing, unexpected = expected - seen, seen - expected
        raise ValueError("%s windows do not exactly cover the frozen evaluation window grid "
                         "(missing=%d, unexpected=%d)" % (fold_id, len(missing), len(unexpected)))
    if data.get("n_files") != len(files) or data.get("n_windows") != len(rows):
        raise ValueError("%s windows file/window counts do not match rows" % fold_id)
    return rows


def _legs(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    values = list(rows)
    if not values:
        raise ValueError("cannot aggregate an empty row set")
    n_windows = len(values)
    means = {leg: sum(float(row[leg]) for row in values) / n_windows
             for leg in ("rel", "tke", "mvpe")}
    result = {leg: 100.0 / (1.0 + 0.5 * means[leg]) for leg in means}
    numerator = 0.0
    denominator = 0
    for row in values:
        penalty = lambda error: error / (0.5 + error)
        weight = (0.5 * (1.0 - penalty(float(row["rel"]))) +
                  0.3 * (1.0 - penalty(float(row["tke"]))) +
                  0.2 * (1.0 - penalty(float(row["mvpe"]))))
        numerator += weight * float(row["S"])
        denominator += int(row["n"])
    result["sps"] = 100.0 * numerator / denominator
    return result


def validate_bundle(bundle_or_path: Union[str, Path, Mapping[str, Any]],
                    protocol: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Validate all five OOF folds and return official per-axis accuracy legs.

    Evidence paths are resolved relative to the bundle file. Inline windows or
    provenance objects remain acceptable only when accompanied by a SHA-256 of
    their canonical JSON bytes. Every fitted component must name a hashed input
    manifest and hashed output artifact, and its full dependency closure is
    checked against the outer holdout.
    """
    protocol_value = load_protocol() if protocol is None else dict(protocol)
    validate_protocol(protocol_value)
    if isinstance(bundle_or_path, Mapping):
        bundle = dict(bundle_or_path)
        base = Path.cwd()
        bundle_digest = _sha_bytes(_canonical(bundle))
    else:
        bundle_path = Path(bundle_or_path)
        raw = bundle_path.read_bytes()
        bundle = json.loads(raw)
        if not isinstance(bundle, dict):
            raise ValueError("bundle JSON must be an object")
        base = bundle_path.resolve().parent
        bundle_digest = _sha_bytes(raw)
    if bundle.get("protocol_id") != protocol_value["protocol_id"]:
        raise ValueError("bundle protocol_id does not match protocol")
    if bundle.get("protocol_hash") != protocol_hash(protocol_value):
        raise ValueError("bundle protocol_hash does not match frozen protocol/data snapshot")
    if not isinstance(bundle.get("method"), str) or not bundle["method"]:
        raise ValueError("bundle method is required")
    if bundle.get("track") not in ("T1", "T2"):
        raise ValueError("bundle track must be T1 or T2")
    fold_entries = bundle.get("folds")
    if not isinstance(fold_entries, list):
        raise ValueError("bundle folds must be a list")
    expected_folds = {fold["id"]: fold for fold in protocol_value["folds"]}
    ids = [entry.get("fold_id") for entry in fold_entries if isinstance(entry, Mapping)]
    if len(ids) != 5 or len(set(ids)) != 5 or set(ids) != set(expected_folds):
        raise ValueError("bundle must contain every protocol fold exactly once")

    rows_by_axis = {"re": [], "aoa": []}
    fold_legs = {}
    source_hashes = {"windows": {}, "provenance": {}, "artifacts": {},
                     "data_snapshot": DATA_SNAPSHOT_SHA256}
    recipes = []
    for entry in fold_entries:
        if not isinstance(entry, Mapping):
            raise ValueError("bundle fold entries must be objects")
        fold_id = entry["fold_id"]
        fold = expected_folds[fold_id]
        windows, windows_hash, _ = _load_bound_json(
            entry.get("windows"), entry.get("windows_sha256"), base,
            "%s windows" % fold_id)
        provenance, provenance_hash, provenance_location = _load_bound_json(
            entry.get("provenance"), entry.get("provenance_sha256"), base,
            "%s provenance" % fold_id)
        provenance_base = (provenance_location.parent if
                           isinstance(entry.get("provenance"), (str, Path)) else base)
        artifact_hashes, recipe = _validate_provenance(
            provenance, provenance_base, fold, bundle)
        rows = _window_rows(windows, fold, protocol_value)
        rows_by_axis[fold["axis"]].extend(rows)
        fold_legs[fold_id] = _legs(rows)
        source_hashes["windows"][fold_id] = windows_hash
        source_hashes["provenance"][fold_id] = provenance_hash
        source_hashes["artifacts"][fold_id] = artifact_hashes
        recipes.append(recipe)

    axis_legs = {axis: _legs(rows) for axis, rows in rows_by_axis.items()}
    return {
        "protocol_id": bundle["protocol_id"],
        "protocol_hash": protocol_hash(protocol_value),
        "method": bundle["method"],
        "track": bundle["track"],
        "axis_legs": axis_legs,
        "fold_legs": fold_legs,
        "source_hashes": source_hashes,
        "bundle_hash": bundle_digest,
        "recipe_hash": _sha_bytes(_canonical(recipes)),
    }


__all__ = ["load_protocol", "validate_protocol", "protocol_hash", "validate_bundle",
           "load_data_snapshot", "expected_eval_windows"]
