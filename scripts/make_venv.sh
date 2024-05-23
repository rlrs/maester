#!/bin/bash
# Important: should be run in the `rocm/pytorch`` container
set -euxo pipefail
export LC_ALL=C

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
GIT_ROOT=$(git rev-parse --show-toplevel)

# cd ${SCRIPT_DIR}
uv venv -p /opt/conda/envs/py_3.10/bin/python3 --seed
source .venv/bin/activate

# get latest rocm6.0 nightly pytorch
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm6.0 --upgrade
uv cache clean # torch takes up the entire home quota
uv pip install -e .
uv pip install torchdata --pre --index-url https://download.pytorch.org/whl/nightly/cpu # nightly torchdata is needed

# Install flash attention
TMP_DIR=$(mktemp -d)
git clone --recurse-submodules https://github.com/ROCmSoftwarePlatform/flash-attention ${TMP_DIR}
cd ${TMP_DIR}

export GPU_ARCHS="gfx90a"
export MAX_JOBS=12 # be nice on the login nodes

# export PYTHON_SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])') # this is for older versions of pytorch
# patch "${PYTHON_SITE_PACKAGES}/torch/utils/hipify/hipify_python.py" hipify_patch.patch
python3 setup.py install