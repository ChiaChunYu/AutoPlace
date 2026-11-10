#!/usr/bin/env bash
# Build the compiled modules: gp_cuda*.so (the CUDA kernels) and def_io*.so (the DEF reader / writer). About 1 min.
#   [PYTHON=python] [NVCC=nvcc] [CXX=g++] [CUDA_ARCH=86] cpp_to_py/build.sh
# CUDA_ARCH: compute capability the kernels are compiled for, default = GPU 0 of this machine (86 = RTX 30xx).
# The kernel file needs no torch header, so any nvcc works; the torch libraries and the CUDA runtime are the ones already
# loaded in the process (linked by name, no absolute path stored: the folder can be moved without rebuilding).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"; NVCC="${NVCC:-nvcc}"; CXX="${CXX:-g++}"
q() { "${PYTHON}" -c "import sysconfig, torch; from torch.utils.cpp_extension import include_paths; print($1)"; }
INC="$(q "' '.join('-I' + p for p in include_paths() + [sysconfig.get_paths()['include']])")"
ABI="$(q "int(torch._C._GLIBCXX_USE_CXX11_ABI)")"; EXT="$(q "sysconfig.get_config_var('EXT_SUFFIX')")"
ARCH="${CUDA_ARCH:-$(q "'%d%d' % torch.cuda.get_device_capability(0)")}"
mkdir -p build
"${NVCC}" -O3 --use_fast_math -arch="sm_${ARCH}" -ccbin "${CXX}" -Xcompiler -fPIC -c gp_cuda_kernel.cu -o build/gp_cuda_kernel.o
"${CXX}" -O3 -fPIC -std=c++17 -fvisibility=hidden -D_GLIBCXX_USE_CXX11_ABI="${ABI}" ${INC} -c gp_cuda.cpp -o build/gp_cuda.o
LIB="$(q "torch.utils.cpp_extension.library_paths()[0]")"
"${CXX}" -shared build/gp_cuda.o build/gp_cuda_kernel.o -L"${LIB}" -ltorch_python -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
    -o "gp_cuda${EXT}"
"${CXX}" -O3 -fPIC -shared -std=c++17 -fvisibility=hidden ${INC} def_io.cpp -o "def_io${EXT}"
echo "built cpp_to_py/gp_cuda${EXT} cpp_to_py/def_io${EXT}"
