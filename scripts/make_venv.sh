#!/bin/bash
# Important: should be run in the `rocm/pytorch`` container
set -euxo pipefail
export LC_ALL=C

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
GIT_ROOT=$(git rev-parse --show-toplevel)

cd ${SCRIPT_DIR}
uv venv -p /opt/conda/envs/py_3.10/bin/python3 --seed
source .venv/bin/activate

# get latest rocm5.7 nightly pytorch
uv pip install torch --index-url https://download.pytorch.org/whl/nightly/rocm5.7 --upgrade
uv pip install -e .