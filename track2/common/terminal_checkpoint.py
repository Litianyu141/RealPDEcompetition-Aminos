"""Save both terminal weight policies without selecting on validation scores."""
import os
import tempfile
from pathlib import Path

import torch


def save_terminal_pair(model, ema, raw_meta, ema_meta, checkpoint_path):
    path = Path(checkpoint_path)
    paths = {k: path.with_name(path.stem + "_terminal_" + k + path.suffix)
             for k in ("raw", "ema")}
    for p in paths.values():
        if p.exists():
            raise FileExistsError(p)
    if raw_meta.get("weights") != "raw" or ema_meta.get("weights") != "ema":
        raise ValueError("terminal metadata must identify RAW and EMA respectively")
    if raw_meta.get("gstep") != ema_meta.get("gstep"):
        raise ValueError("RAW and EMA must come from the same terminal step")
    raw = model.state_dict()
    if set(raw) != set(ema):
        raise ValueError("EMA state keys differ from the live model")
    path.parent.mkdir(parents=True, exist_ok=True)
    for which, meta in (("raw", raw_meta), ("ema", ema_meta)):
        state = raw if which == "raw" else ema
        copied = {k: v.detach().to(dtype=raw[k].dtype).cpu().clone() for k, v in state.items()}
        final_meta = dict(meta, selection="fixed_terminal_" + which)
        handle, tmp_name = tempfile.mkstemp(prefix=paths[which].name + ".", dir=path.parent)
        os.close(handle)
        try:
            torch.save({"model_state_dict": copied, "meta": final_meta}, tmp_name)
            os.link(tmp_name, paths[which])  # atomic install that refuses to replace another run
        finally:
            os.unlink(tmp_name)
    return paths
