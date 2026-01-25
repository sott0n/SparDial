# SparDial Python Package


def spardial_jit(*args, **kwargs):
    """Lazy wrapper for spardial_jit (NumPy or PyTorch)."""
    from spardial.backend import spardial_jit as _spardial_jit

    return _spardial_jit(*args, **kwargs)


__all__ = ["spardial_jit"]
