#!/usr/bin/env python3
"""Bind SBAND2's unchanged Re3 OOF proof to freshly trained teacher outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import band_teachers as bt

ROOT = bt.ROOT
ORIGINAL = ROOT / "Track1/U-net/VariantFiveFoldEvaluation/config/campaign.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = (ROOT / a.work).resolve()
    assert work.is_relative_to(Path(__file__).parent / "Logger") or work.is_relative_to(Path(os.environ.get("RPDE_FINAL_OUTPUT", "/output"))), work
    spec = json.loads(bt.SPEC.read_text())
    campaign = json.loads(ORIGINAL.read_text())
    index = {j["id"]: j for j in campaign["jobs"]}
    expected = {}
    for job in bt.jobs(spec):
        name, fold, role = job["name"], job["fold"], job["role"]
        jid = fold + "__outer__LIVE__" + role
        assert jid in index
        cp = work / "logs" / (name + ".complete.json")
        done = json.loads(cp.read_text())
        output = bt.output(work, name)
        assert done["status"] == "OK" and done["node"] == name and bt.sha(output) == done["output_sha256"]
        index[jid]["output"] = str(output)
        expected[jid] = "decision_band_" + name
        cfg = index[jid]["config"]
        assert cfg["seed"] == 0 and cfg["fold"] == 0
        assert set(cfg["stages"][1]["sources"][0]["keep_files"]) == \
            {Path(s).name for s in index[jid]["train_files"]}
    assert len(expected) == 12
    for filename, data in (("ship_campaign.json", campaign), ("expected_variants.json", expected)):
        p = work / filename
        text = json.dumps(data, indent=2, sort_keys=True) + "\n"
        if p.exists():
            assert p.read_text() == text, "campaign binding drift " + filename
        else:
            p.write_text(text)
    print("SHIP_CAMPAIGN_BOUND", len(expected), bt.sha(work / "ship_campaign.json"))


if __name__ == "__main__":
    main()
