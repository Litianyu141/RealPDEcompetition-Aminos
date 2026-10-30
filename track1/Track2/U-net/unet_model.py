"""SUPERSEDED 2026-07-31 -- do not import, do not ship.

This was the original 2026-07-23 baseline U-net.  Track 2 bundles used to ship THIS
file while their checkpoints were trained with Track1/U-net/unet_model.py, which by
2026-07-30 had grown build_from_meta, in_norm, in_aug and the fuse branches.  The
mismatch was silent because the base architecture never changed and the stale module
still strict-loaded -- and a parameter-free architecture key would therefore have been
dropped without any error at all.

There is ONE model definition in this repo and it lives at Track1/U-net/unet_model.py.
build_t2_bundle.py copies that one.  Import it from there.
"""
raise ImportError(
    "Track2/U-net/unet_model.py is superseded; import Track1/U-net/unet_model.py "
    "-- it is the definition every checkpoint in this repo was trained with."
)
