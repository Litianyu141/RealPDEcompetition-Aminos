#!/usr/bin/env python3
"""Add NEUTRAL activity keys to a from-scratch band calib so the bootstrap bundle passes the
builder's _KCELL loudness assert.  k_cell=1 and p=0 make the activity multiplier exactly 1.0,
so the band is BIT-IDENTICAL -- this bridges stage 6 to stage 7, which fits the real keys."""
import sys
import numpy as np
z = dict(np.load(sys.argv[1], allow_pickle=True))
z.setdefault("k_cell", np.ones((20, 3), np.float32))
z.setdefault("act_const", np.ones(3, np.float32))
z.setdefault("p", np.float32(0.0))
z.setdefault("lam", np.float32(1.0))
z.setdefault("self_norm", np.bool_(True))
z.setdefault("act_kind", np.str_("std"))
np.savez(sys.argv[2], **z)
print(f"boot calib: {sys.argv[1]} + neutral activity keys -> {sys.argv[2]}")
