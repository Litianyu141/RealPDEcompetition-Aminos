"""Final-phase Track 2 source-only training and v156g-family TTT adapter."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def train(data_dir, output_dir, device):
    here = Path(__file__).resolve()
    # In the repo this file sits 3 levels below the root; in the Final container it is /workspace/solution.py,
    # which has only 2 parents, so parents[3] must not be indexed unguarded.
    path = here.parents[3] / "Track1/U-net/VariantDecisionRepro/final_pipeline.py" if len(here.parents) > 3 else None
    if path is None or not path.is_file():
        path = here.parent / "final_pipeline.py"
    spec = importlib.util.spec_from_file_location("realpde_final_pipeline_t2", path)
    if spec is None or spec.loader is None:
        raise ImportError("Final pipeline source missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.train("T2", data_dir, output_dir, device)


def get_ttt_model(output_dir, device):
    if str(device) != "cuda":
        raise ValueError("v156g final harness specifies device='cuda'")
    folder = Path(output_dir).resolve()
    need = ("submission.py", "unet_model.py", "ttt_model.py", "vgain_head.py", "sband_head.py",
            "model.pth", "model2.pth", "vgain.pth", "sband.pth", "calib.npz", "ll_p1.npy")
    missing = [n for n in need if not (folder / n).is_file()]
    if missing:
        raise FileNotFoundError("T2 output missing: " + ", ".join(missing))
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("realpde_t2_decision_runtime", folder / "submission.py")
    if spec is None or spec.loader is None:
        raise ImportError("could not load T2 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_ttt_model(str(folder), device)
