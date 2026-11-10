"""Compiled modules: gp_cuda (CUDA kernels) and def_io (DEF reader / writer). Each is None when it is not built: the
placer then uses its PyTorch operators / its Python reader (run_placement.py logs which ones are in use)."""
import torch  # noqa: F401  gp_cuda links the torch libraries by name: they must be loaded first

try:
    from . import gp_cuda
except ImportError:
    gp_cuda = None
try:
    from . import def_io
except ImportError:
    def_io = None
