#!/usr/bin/env python3
"""Derive a reproduction chain's anchors from the LIVE CARD, by MD5 against the shipped zip.

WHY THIS EXISTS -- defect class D6, fourth recurrence (2026-08-10, -08-16, -08-18, -08-19).
Each reproduce_*.sh carried a hand-maintained SNAPSHOT of the live card: template, head, calib,
sband, four acceptance anchors, plus a hard-coded LIVE= path for the md5 guard to compare against.
Nine independent values.  Every time the card moved, some drifted and some did not, and the
dangerous mode is the SILENT PASS: on 2026-08-19 reproduce_t1.sh still named the v70 artifacts, so
a rebuild that reproduced the stale sps 54.950 would print ACCEPT while the LIVE card's 55.314 sat
0.064 outside the same window.  Three of four anchors matched, because PCDEL was a band-only
change -- "when only ONE of four anchors can move, the other three agreeing proves nothing."

THE FIX IS NOT A FRESHER SNAPSHOT.  All nine values are derived from ONE fact -- the card's name --
and every derived artifact is then verified BY MD5 against the corresponding member inside the
card's own zip.  Consequences:
  * moving the card is a two-line edit of submissions/LIVE_CARD (name + zip md5) that cannot be
    partially applied, and the chain refuses if those two lines disagree with the zip on disk;
  * an artifact is located by CONTENT, not by name.  That is how the live T1 sband head is found
    at sband/sband_xcb.pth even though the report records the wipeable scratch path it was built
    in (a host scratch directory);
  * the SECOND blend member is derived too -- both live cards are 2-member cross-recipe blends and
    neither chain mentioned a second forecaster at all.

usage:
  live_card.py --track 1 --emit shell      # eval-able assignments for reproduce_t1.sh
  live_card.py --track 2 --emit json
  live_card.py --track 1 --write t1_v71_xcb   # (re)generate the pointer for a new card
"""
import argparse
import hashlib
import json
import pathlib
import sys
import zipfile

R = pathlib.Path(__file__).resolve().parents[1]
TRACK_DIR = {1: R / "Track1/U-net", 2: R / "Track2/U-net"}
# DPBOOTSTRAP (2026-09-12, docs/decision/RECIPE.md open item 5): the committed record of
# the live cards' IDENTITY -- name, zip md5, four accuracy anchors, the T2 ONLINE dict, the template
# path and every artifact's md5.  Numbers and hashes, not weights, so it satisfies the official
# training-code-only requirement.  It exists because Track*/submissions/ is gitignored: on a clean
# clone, and in the organizers' Decision-Phase environment, there is NO live zip, and both chains died
# at their `eval $(live_card.py --emit shell)` line before training a single stage.
ANCHORS = R / "docs/decision/live_card_anchors.json"
# where an artifact of each kind may legitimately live in-repo, searched by CONTENT
SEARCH = {
    1: ["checkpoints/t1_u-net", "Track1/U-net/sband", "Track1/U-net"],
    2: ["checkpoints/t1_u-net", "Track1/U-net/sband", "Track2/U-net", "Track1/U-net"],
}
# zip member  ->  report key that should name it
MEMBER_OF = {
    # 2026-08-26 audit: model_2..model_7 added.  Before this MEMBER_OF[1] stopped at model_1.pth,
    # so a shipped 3-member card was resolved, md5-checked and REBUILT as the 2-member card and the
    # chain still printed REPRODUCE_T1_OK -- the 3-vs-2 accuracy difference (+0.005 Final) is far
    # inside the +-0.3 sps acceptance window, so the anchors physically could not see the loss.
    # Members absent from a zip are reported as None by the existing `if member not in members`
    # branch, so this is a no-op for the 2-member card that ships today.
    1: {"model.pth": "checkpoint", "model_1.pth": "extra_checkpoints[0]",
        "model_2.pth": "extra_checkpoints[1]", "model_3.pth": "extra_checkpoints[2]",
        "model_4.pth": "extra_checkpoints[3]", "model_5.pth": "extra_checkpoints[4]",
        "model_6.pth": "extra_checkpoints[5]", "model_7.pth": "extra_checkpoints[6]",
        "vgain.pth": "head", "calib.npz": "calib", "sband.pth": "sband"},
    2: {"model.pth": "checkpoint", "model2.pth": "checkpoint2",
        # 2026-08-23 SBPORT (t2_v49): the T2 card ships the Track-1 sband head too
        "vgain.pth": "head", "calib.npz": "calib", "sband.pth": "sband"},
}


# RELAYOUT 2026-09-07: cards and LIVE_CARD live at the TRACK root (Track{1,2}/submissions, Track{1,2}/LIVE_CARD);
# tdir is still the method dir (Track*/U-net), so every reference goes through tdir.parent.  The old paths are symlinks.
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def die(msg):
    print("LIVE_CARD_REFUSE: " + msg, file=sys.stderr)
    sys.exit(1)


def find_by_md5(track, want, hint="", size=None, extra=()):
    """Locate an in-repo file whose CONTENT hashes to `want`.  Prefer the recorded path when it
    still resolves in-repo; otherwise search the plausible dirs.  Returns (path, note).

    SIZE PREFILTER: md5 cannot match unless the byte length does, and these dirs hold hundreds of
    30 MB checkpoints -- hashing them all took minutes.  Filtering on st_size first cuts the
    candidate set to a handful and is exact (never skips a true match)."""
    # The recorded path wins ONLY if it is inside the repo.  The live T1 sband head was built in
    # a host scratch directory -- WIPEABLE SCRATCH -- and the report still names it there, so
    # preferring the hint unconditionally would hand the chain a path that can vanish and would
    # defeat the whole point of rescuing the file into sband/.  In-repo always wins.
    _scratch = None
    if hint:
        hp = pathlib.Path(hint)
        if not hp.is_absolute():
            hp = TRACK_DIR[track] / hint
        if hp.is_file() and md5(hp) == want:
            if str(hp).startswith(str(R) + "/"):
                return hp, ""
            _scratch = hp
    for d in SEARCH[track]:
        dd = R / d
        if not dd.is_dir():
            continue
        for p in sorted(dd.iterdir()):
            if not p.is_file():
                continue
            if size is not None and p.stat().st_size != size:
                continue
            if md5(p) == want:
                return p, "relocated by md5 (report path did not resolve in-repo)"
    # 2026-09-08 (clean-clone dry run): the CARD DIRECTORY Track*/submissions/<card>/ is the one
    # place every zip member is guaranteed to exist on a clone (the card dir
    # travels with the reproduction package; server-side checkpoints/ do not).  Searched LAST so an
    # in-repo checkpoint still wins when present.
    for dd in extra:
        dd = pathlib.Path(dd)
        if not dd.is_dir():
            continue
        for q in sorted(dd.iterdir()):
            if q.is_file() and (size is None or q.stat().st_size == size) and md5(q) == want:
                return q, "found in the card directory (no in-repo checkpoint copy on this box)"
    if _scratch is not None:
        return _scratch, ("ONLY COPY IS OUT-OF-REPO AND WIPEABLE: %s -- rescue it into the repo "
                          "and git-add it, or a Decision-Phase rebuild cannot run" % _scratch)
    return None, "NOT FOUND IN REPO"


def load_pointer(track, tdir):
    # NOT under submissions/ : .gitignore:281 ignores **/submissions/, so a pointer there could only
    # be tracked with `git add -f` and would be invisible to a fresh clone -- the exact failure this
    # file exists to prevent.  It lives at the track's U-net root, which is tracked normally.
    ptr = tdir.parent / "LIVE_CARD"
    if not ptr.is_file():
        die("no pointer file at %s -- create it with --write <bundle-name>" % ptr)
    kv = {}
    for line in ptr.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    for k in ("name", "zip_md5"):
        if not kv.get(k):
            die("%s lacks a '%s =' line" % (ptr, k))
    return kv, ptr


def write_pointer(track, name):
    tdir = TRACK_DIR[track]
    z = tdir.parent / "submissions" / (name + ".zip")
    if not z.is_file():
        die("no such zip: %s" % z)
    ptr = tdir.parent / "LIVE_CARD"
    keep = ""
    if ptr.is_file():
        for line in ptr.read_text().splitlines():
            if line.split("#", 1)[0].strip().startswith("template"):
                keep = line.rstrip() + "\n"
    ptr.write_text(
        "# The bundle currently on the Codabench leaderboard for Track %d.\n"
        "# reproduce_t%d.sh derives EVERY artifact and all four acceptance anchors from this one\n"
        "# name, by md5 against the zip below (see common/live_card.py).  Change both lines\n"
        "# together when the card moves; the chain refuses if they disagree with the zip on disk.\n"
        "name    = %s\nzip_md5 = %s\n" % (track, track, name, md5(z)) + keep)
    print("wrote %s -> %s  %s" % (ptr, name, md5(z)))


def resolve(track, card=None, quiet=False):
    tdir = TRACK_DIR[track]
    if card is None:
        kv, ptr = load_pointer(track, tdir)
        card = kv["name"]
        want_zip = kv["zip_md5"]
    else:
        kv, ptr, want_zip = {}, None, None

    zp = tdir.parent / "submissions" / (card + ".zip")
    rp = tdir.parent / "submissions" / card / "verify_report.json"
    if not zp.is_file():
        die("card zip missing: %s" % zp)
    if not rp.is_file():
        die("card has no verify_report.json: %s" % rp)
    got_zip = md5(zp)
    if want_zip and got_zip != want_zip:
        die("%s says zip_md5 = %s but %s hashes to %s.  The pointer and the zip disagree; one of "
            "them is stale.  Re-run with --write %s if the zip is the truth." %
            (ptr, want_zip, zp.name, got_zip, card))

    rep = json.loads(rp.read_text())
    out = {"name": card, "zip": str(zp), "zip_md5": got_zip,
           "subscores": rep["subscores"], "notes": []}

    # ---- artefacts, each located BY CONTENT against the zip member it must equal
    with zipfile.ZipFile(zp) as z:
        members = set(z.namelist())
        for member, key in MEMBER_OF[track].items():
            if member not in members:
                out.setdefault("artifacts", {})[key] = None
                continue
            _blob = z.read(member)
            want, _sz = md5_bytes(_blob), len(_blob)
            del _blob
            if key.endswith("[0]"):
                base = key[:-3]
                hint = (rep.get(base) or [""])[0] if isinstance(rep.get(base), list) else ""
            else:
                hint = rep.get(key) or ""
            p, note = find_by_md5(track, want, hint, size=_sz, extra=[tdir.parent / "submissions" / card])
            if p is None:
                die("zip member %s (md5 %s) has NO counterpart in the repo.  The chain cannot "
                    "rebuild what it does not contain -- rescue the file and git-add it. "
                    "(report recorded: %s)" % (member, want, hint or "nothing"))
            out.setdefault("artifacts", {})[key] = str(p)
            if note:
                out["notes"].append("%s: %s" % (key, note))
            if not quiet:
                print("  [md5] %-14s <- %-52s %s" % (member, str(p).replace(str(R) + "/", ""), want[:8]))

    # ---- the template is RENDERED into the zip (placeholders substituted), so it cannot be
    #      md5-matched; verify existence only and say so.
    tpl = rep.get("template", "")
    src = "verify_report.json"
    if not tpl:
        # Pre-2026-08-19 bundles: build_t2_bundle.py did not record the template at all.  Take it
        # from an explicit `template =` line in the pointer -- but do NOT trust that line: it is
        # VERIFIED below by rendering-equivalence against the zip's own submission.py, so a wrong
        # pointer line is caught rather than believed.
        tpl = kv.get("template", "")
        src = "the pointer file (human-supplied)"
        if not tpl:
            die("verify_report.json for %s records no `template`, and the pointer has no "
                "`template =` line to supply it.  Either rebuild the card with the patched "
                "builder (2026-08-19), or add a `template =` line to %s (it will be verified "
                "against the zip, not trusted)." % (card, ptr))
    # a recorded template may be absolute, track-relative ("./submission_t2_v47_fp16.py", as the
    # T2 builder writes calib) or repo-relative ("Track2/U-net/..."); try all three
    # 2026-09-08 (clean-clone run on the SAME box as the production clone): the absolute path recorded
    # in the report resolved to the OTHER clone and the tracked-in-git guard failed on it.  In-repo
    # candidates go FIRST; the recorded absolute path is only a last resort.  Every candidate is still
    # VERIFIED by rendering-equivalence against the zip below.
    _cands = [tdir / pathlib.Path(tpl).name]
    if not pathlib.Path(tpl).is_absolute():
        _cands += [tdir / tpl, R / tpl]
    else:
        _cands.append(pathlib.Path(tpl))
    tp = next((c for c in _cands if c.is_file()), None)
    if tp is None:
        die("template named by %s does not exist under any of %s" %
            (src, [str(c) for c in _cands]))

    # VERIFY THE TEMPLATE BY CONTENT.  md5 cannot match -- the builder RENDERS the template into
    # submission.py, substituting __PLACEHOLDER__ tokens (TTT_LR, HALFWIDTH, ONLINE, CHUNK, ...).
    # But rendering only ever rewrites the placeholder LINES, so: same line count, and every
    # differing line must carry a __TOKEN__ in the template.  Measured on t2_v47_fp16: exactly 4
    # lines differ and all 4 are placeholders.  This turns "existence-checked" into a real check
    # and means a wrong pointer line is REFUSED instead of silently reproducing another template.
    with zipfile.ZipFile(zp) as z:
        rendered = z.read("submission.py").decode("utf-8", "replace").splitlines()
    tsrc = tp.read_text().splitlines()
    if len(tsrc) != len(rendered):
        die("template %s has %d lines but the zip's submission.py has %d.  Rendering only rewrites "
            "placeholder LINES, so a line-count change means this is NOT the template that built "
            "%s." % (tp.name, len(tsrc), len(rendered), card))
    bad, nsub = [], 0
    for i, (t, r) in enumerate(zip(tsrc, rendered), start=1):
        if t == r:
            continue
        nsub += 1
        if "__" not in t:
            bad.append(i)
    if bad:
        die("template %s differs from the zip's submission.py on %d line(s) that carry NO "
            "__PLACEHOLDER__ (first: line %d).  Rendering cannot produce that, so this is NOT the "
            "template that built %s." % (tp.name, len(bad), bad[0], card))
    out["template"] = str(tp)
    out["template_source"] = src
    out["notes"].append("template: verified by rendering-equivalence against the zip's "
                        "submission.py (%d placeholder line(s) substituted, 0 other differences); "
                        "named by %s" % (nsub, src))

    # ---- drift detector: name any NEWER card-shaped report than the one we are anchored to
    newer, mine = [], rp.stat().st_mtime
    for d in sorted((tdir.parent / "submissions").iterdir()):
        f = d / "verify_report.json"
        if d.is_dir() and f.is_file() and f.stat().st_mtime > mine and \
                d.name.startswith("t%d_v" % track):
            newer.append(d.name)
    out["newer_card_shaped_reports"] = newer
    return out


def dump_anchors(path=None):
    """Write the committed identity record for BOTH tracks, derived from the zips on this box.

    Run this where the cards exist (Mac or a server) and commit the result.  It is the ONLY place the
    live cards' identity is recorded in tracked form, so a bootstrap run on a zip-less clone can still
    anchor itself and still md5-verify any artifact that happens to be present.
    """
    import datetime
    out = {"schema": "live_card_anchors_v1",
           "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "generated_by": "common/live_card.py --dump-anchors",
           "why": ("the live card zips live under Track*/submissions/, which .gitignore excludes, so a "
                   "clean clone has no zip and live_card.py cannot resolve the card.  The Decision Phase "
                   "accepts training code only, no weights, so the organizers' environment has no zip "
                   "either.  This file records the identity as numbers and hashes: name, zip md5, the "
                   "four accuracy anchors, the T2 ONLINE knob dict, the repo-relative template path, "
                   "and every shipped artifact's md5.  Regenerate with --dump-anchors whenever "
                   "LIVE_CARD moves, and commit it in the same edit."),
           "tracks": {}}
    for track in (1, 2):
        d = resolve(track, quiet=True)
        rep = json.loads((pathlib.Path(d["zip"]).with_suffix("") / "verify_report.json").read_text())
        art = d.get("artifacts") or {}
        out["tracks"][str(track)] = {
            "name": d["name"], "zip_md5": d["zip_md5"],
            "template": str(pathlib.Path(d["template"]).relative_to(R)),
            "anchors": {"rel": d["subscores"]["rel_l2_score"], "tke": d["subscores"]["tke_score"],
                        "mvpe": d["subscores"]["mvpe_score"], "sps": d["subscores"]["sps_score"]},
            "online": rep.get("online"),
            "artifact_md5": {m: md5_bytes(z_read(d["zip"], m)) for m in sorted(MEMBER_OF[track])
                             if z_has(d["zip"], m)},
            "source": {"zip": d["zip"], "verify_report": str(pathlib.Path(d["zip"]).with_suffix("")
                                                             / "verify_report.json"),
                       "notes": d["notes"]},
        }
    p = pathlib.Path(path) if path else ANCHORS
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote %s" % p)
    for t, v in out["tracks"].items():
        print("  T%s %s  zip %s  anchors rel %s tke %s mvpe %s sps %s  artifacts %d"
              % (t, v["name"], v["zip_md5"][:8], v["anchors"]["rel"], v["anchors"]["tke"],
                 v["anchors"]["mvpe"], v["anchors"]["sps"], len(v["artifact_md5"])))
    return out


def z_read(zip_path, member):
    with zipfile.ZipFile(zip_path) as z:
        return z.read(member)


def z_has(zip_path, member):
    with zipfile.ZipFile(zip_path) as z:
        return member in set(z.namelist())


def resolve_bootstrapped(track):
    """The same record `resolve()` returns, rebuilt from the committed anchors file.

    Artifact PATHS are deliberately empty: on a zip-less clone the artifacts do not exist, and every
    use of LIVE_CKPT/LIVE_HEAD/LIVE_CALIB/LIVE_SBAND in both chains is a `${VAR:-$LIVE_*}` default, so
    an empty value forces the caller to have produced them -- which is exactly what REPRO_SCOPED=1 and
    BAND_PIPELINE=1 do.  A bootstrap run that has NOT set those modes therefore fails loudly at the
    first `[ -e "$X" ] || MISSING` check instead of silently rebuilding the wrong card.
    """
    if not ANCHORS.is_file():
        die("BOOTSTRAP needs %s, which is missing.  Regenerate it on a box that holds the live card "
            "zips: python3 common/live_card.py --track %d --dump-anchors" % (ANCHORS, track))
    rec = json.loads(ANCHORS.read_text())["tracks"].get(str(track))
    if rec is None:
        die("BOOTSTRAP: %s has no track %d record" % (ANCHORS, track))
    tp = R / rec["template"]
    if not tp.is_file():
        die("BOOTSTRAP: the recorded template %s is not in this clone" % rec["template"])
    return {"name": rec["name"], "zip": "", "zip_md5": rec["zip_md5"],
            "subscores": {"rel_l2_score": rec["anchors"]["rel"], "tke_score": rec["anchors"]["tke"],
                          "mvpe_score": rec["anchors"]["mvpe"], "sps_score": rec["anchors"]["sps"]},
            "artifacts": {}, "template": str(tp), "online": rec.get("online"),
            "artifact_md5": rec.get("artifact_md5", {}),
            "newer_card_shaped_reports": [],
            "notes": ["BOOTSTRAP: identity read from the committed %s, not from a zip.  No artifact "
                      "path is emitted, so the caller must be in a mode that PRODUCES them "
                      "(REPRO_SCOPED=1, BAND_PIPELINE=1); the md5-against-zip guard cannot run and "
                      "must be skipped." % ANCHORS.relative_to(R)]}


def emit_shell(d, bootstrapped=False):
    """The one place the eval-able assignments are printed, so bootstrap and zip paths cannot drift."""
    art = d.get("artifacts") or {}

    def g(k):
        return art.get(k) or ""
    print("LIVE_NAME=%s" % d["name"])
    print("LIVE_ZIP_MD5=%s" % d["zip_md5"])
    print("LIVE_CKPT=%s" % g("checkpoint"))
    print("LIVE_CKPT2=%s" % (g("extra_checkpoints[0]") or g("checkpoint2")))
    print("LIVE_HEAD=%s" % g("head"))
    print("LIVE_CALIB=%s" % g("calib"))
    print("LIVE_SBAND=%s" % g("sband"))
    print("LIVE_TEMPLATE=%s" % d["template"])
    _onl = d.get("online")
    if _onl is None and not bootstrapped:
        # T2 (2026-09-09, DP-CHAIN-2GEN): the ONLINE knob dict the live card was built with, read from
        # the card directory's verify_report.json.  Empty when absent (T1 cards have none).
        try:
            _rep = pathlib.Path(d["zip"]).with_suffix("") / "verify_report.json"
            _onl = json.loads(_rep.read_text()).get("online") if _rep.exists() else None
        except Exception:
            _onl = None
    print("LIVE_ONLINE='%s'" % json.dumps(_onl, separators=(",", ":")) if _onl else "LIVE_ONLINE=")
    sub = d["subscores"]
    print("LIVE_REL=%s" % sub["rel_l2_score"])
    print("LIVE_TKE=%s" % sub["tke_score"])
    print("LIVE_MVPE=%s" % sub["mvpe_score"])
    print("LIVE_SPS=%s" % sub["sps_score"])
    for n in d.get("notes", []):
        print("LIVE_NOTE=%r" % n)
    if d.get("newer_card_shaped_reports"):
        print("LIVE_DRIFT=%s" % ",".join(d["newer_card_shaped_reports"]))


IDENTITY_FIELDS = ("LIVE_NAME", "LIVE_ZIP_MD5", "LIVE_TEMPLATE", "LIVE_ONLINE",
                   "LIVE_REL", "LIVE_TKE", "LIVE_MVPE", "LIVE_SPS")


def _parse_emit(lines):
    kv = {}
    for ln in lines:
        if "=" not in ln or ln.startswith("LIVE_NOTE"):
            continue
        k, _, v = ln.partition("=")
        kv[k] = v
    return kv


def verify_bootstrap(track):
    """Witness that the committed anchors file still agrees with the zip on this box.

    Run it wherever the zips exist, before trusting a bootstrap run elsewhere.  It compares the two
    emits field by field, because a silently stale anchors file is the exact failure mode this whole
    module was written to prevent (a hand-maintained snapshot that drifted and then PASSED).
    """
    import io
    import contextlib
    outs = {}
    for label, boot in (("zip", False), ("bootstrap", True)):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                d = resolve_bootstrapped(track) if boot else resolve(track, quiet=True)
                emit_shell(d, bootstrapped=boot)
        except SystemExit as e:
            print("VERIFY_BOOTSTRAP_FAIL -- %s emit exited: %s" % (label, e))
            return 1
        outs[label] = _parse_emit(buf.getvalue().splitlines())
    bad = []
    for f in IDENTITY_FIELDS:
        z, b = outs["zip"].get(f), outs["bootstrap"].get(f)
        if f == "LIVE_TEMPLATE":
            # the zip path resolves through the card's own dir first; compare the file NAME, which is
            # what identifies the template, and require both to exist
            z = pathlib.Path(z or "").name
            b = pathlib.Path(b or "").name
        if z != b:
            bad.append("%s: zip=%r bootstrap=%r" % (f, z, b))
        print("  %-14s %-46s %s" % (f, str(z)[:46], "OK" if not bad or bad[-1].split(":")[0] != f else "DIFFERS"))
    # artifact paths MUST be empty under bootstrap: that is what forces a producing mode
    for f in ("LIVE_CKPT", "LIVE_CKPT2", "LIVE_HEAD", "LIVE_CALIB", "LIVE_SBAND"):
        if outs["bootstrap"].get(f):
            bad.append("%s: bootstrap must emit an empty path, got %r" % (f, outs["bootstrap"][f]))
    if bad:
        print("VERIFY_BOOTSTRAP_FAIL")
        for b in bad:
            print("    " + b)
        return 1
    print("VERIFY_BOOTSTRAP_OK  track %d: %s identity matches the zip on this box, and no artifact "
          "path leaks into the bootstrap emit" % (track, outs["bootstrap"]["LIVE_NAME"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, required=True, choices=(1, 2))
    ap.add_argument("--emit", default="json", choices=("json", "shell"))
    ap.add_argument("--card", default="", help="override the pointer (diagnostics)")
    ap.add_argument("--write", default="", help="(re)generate the pointer for this bundle name")
    ap.add_argument("--check", default="",
                    help="comma-separated member=path pairs to VERIFY against the card's zip, e.g. "
                         "model.pth=...,vgain.pth=... .  Refuses on any md5 mismatch.  This is the "
                         "guard: it must PASS for the live quintuple and REFUSE if any single "
                         "artifact is swapped for a previous card's.")
    ap.add_argument("--dump-anchors", action="store_true",
                    help="write docs/decision/live_card_anchors.json from the zips on this box")
    ap.add_argument("--verify-bootstrap", action="store_true",
                    help="witness: the bootstrap emit must equal the zip-derived emit on every field "
                         "that carries identity (name, zip md5, template, ONLINE, four anchors)")
    ap.add_argument("--bootstrap", action="store_true",
                    help="resolve the live card from the committed anchors file instead of the zip, "
                         "for a clean clone / Decision-Phase environment that ships no weights")
    a = ap.parse_args()
    if a.write:
        write_pointer(a.track, a.write)
        return
    if a.dump_anchors:
        dump_anchors()
        return
    if a.verify_bootstrap:
        sys.exit(verify_bootstrap(a.track))
    if a.check and a.bootstrap:
        # The guard's whole authority is "the zip is the truth".  With no zip there is no truth to
        # compare against, and pretending otherwise is how a stale pair once passed while the live card
        # failed (the defect this file exists to prevent).  Refuse loudly; the chains skip --check
        # under BOOTSTRAP=1 and fall back to the accuracy anchors, which are committed numbers.
        print("LIVE_CARD_GUARD_SKIPPED -- BOOTSTRAP has no zip to md5 against.  Identity is the "
              "committed anchors file and acceptance is the four accuracy anchors, not a hash.",
              file=sys.stderr)
        sys.exit(0)
    if a.check:
        # THE GUARD.  Compare each supplied path against the corresponding member INSIDE the live
        # card's zip.  The old chains compared 2 of 6 artifacts against a hard-coded LIVE= snapshot,
        # which is how a stale pair passed while the live card failed.  Here the zip is the truth
        # and every member the chain will package is checked.
        tdir = TRACK_DIR[a.track]
        kv, _ = load_pointer(a.track, tdir)
        zp = tdir.parent / "submissions" / (kv["name"] + ".zip")
        bad = []
        with zipfile.ZipFile(zp) as z:
            names = set(z.namelist())
            for pair in a.check.split(","):
                if not pair.strip():
                    continue
                member, path = pair.split("=", 1)
                member, path = member.strip(), path.strip()
                if member not in names:
                    bad.append("%s: not a member of %s" % (member, zp.name))
                    continue
                if not path:
                    bad.append("%s: no path supplied (the chain resolved it to empty)" % member)
                    continue
                if not pathlib.Path(path).is_file():
                    bad.append("%s: %s does not exist" % (member, path))
                    continue
                want, got = md5_bytes(z.read(member)), md5(path)
                ok = (want == got)
                print("  [guard] %-14s %-52s %s %s" %
                      (member, pathlib.Path(path).name, got[:8], "OK" if ok else "!= " + want[:8]))
                if not ok:
                    bad.append("%s: %s hashes to %s but the card ships %s"
                               % (member, path, got, want))
        if bad:
            print("LIVE_CARD_GUARD_FAIL -- the chain would rebuild something the card is NOT:",
                  file=sys.stderr)
            for b in bad:
                print("    " + b, file=sys.stderr)
            sys.exit(1)
        print("LIVE_CARD_GUARD_OK  every checked artifact matches %s" % zp.name)
        return
    quiet = (a.emit == "shell")
    d = resolve_bootstrapped(a.track) if a.bootstrap else resolve(a.track, a.card or None, quiet=quiet)
    art, sub = d["artifacts"], d["subscores"]
    if a.emit == "json":
        print(json.dumps(d, indent=1))
        return
    emit_shell(d, bootstrapped=a.bootstrap)


if __name__ == "__main__":
    main()
