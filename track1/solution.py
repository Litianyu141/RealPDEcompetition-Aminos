"""Final-phase Track 1 source-only training and v185-family inference adapter."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def train(data_dir, output_dir, device):
    path = Path(__file__).resolve().parent / "final_pipeline.py"
    spec = importlib.util.spec_from_file_location("realpde_final_pipeline_t1", path)
    if spec is None or spec.loader is None:
        raise ImportError("Final pipeline source missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.train("T1", data_dir, output_dir, device)


def _runtime(output_dir):
    folder = Path(output_dir).resolve()
    need = ("submission.py", "unet_model.py", "vgain_head.py", "sband_head.py",
            "model.pth", "model_1.pth", "vgain.pth", "sband.pth", "calib.npz", "mean_std_real.pt")
    missing = [n for n in need if not (folder / n).is_file()]
    if missing:
        raise FileNotFoundError("T1 output missing: " + ", ".join(missing))
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("realpde_t1_decision_runtime", folder / "submission.py")
    if spec is None or spec.loader is None:
        raise ImportError("could not load T1 runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Predictor:
    def __init__(self, module):
        self._module = module

    def predict(self, input_array):
        return self._module.predict(input_array, metadata={})


def load_predictor(output_dir, device):
    if str(device) != "cuda":
        raise ValueError("v185 was trained and served with CUDA; final harness specifies device='cuda'")
    return Predictor(_runtime(output_dir))
