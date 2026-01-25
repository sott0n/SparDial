# SparDial Python Package

# Avoid eager imports to prevent circular import with spardial.ir.
def spmv(*args, **kwargs):
    """Lazy wrapper for NumPy/SciPy SpMV API."""
    from spardial.numpy_backend import spmv as _spmv
    return _spmv(*args, **kwargs)

__all__ = ["spmv"]
