#!/usr/bin/env bash
# Creates the `ect` mamba environment for the ESA Climate Toolbox.
#
# Why a separate env: `uv pip install esa-climate-toolbox` into .venv fails because
# its GDAL dependency needs the system libgdal, which pip cannot provide. conda-forge
# ships it. build_gee.py / build_ect.py / targets_ect.py never import torch, so an
# ECT dataset build runs entirely in this env, and train/evaluate stay in .venv.
#
# Usage (run once):    bash setup/default/ect_env.sh
# Then per session:    mamba activate ect        (instead of the .venv activation)
#                      python -m src.dataset.targets_ect --ecv BIOMASS

set -e
MAMBA_BASE="$HOME/miniforge3"
export PATH="$MAMBA_BASE/bin:$PATH"
ENV_NAME="ect"

if mamba env list | grep -qE "^$ENV_NAME[[:space:]]"; then
    echo "env '$ENV_NAME' already exists; to rebuild: mamba env remove -n $ENV_NAME"
    exit 0
fi

mamba create -n "$ENV_NAME" -c conda-forge -y \
    python=3.11 esa-climate-toolbox earthengine-api tifffile rasterio requests tqdm matplotlib

echo
echo "Done. Activate with:  mamba activate $ENV_NAME"
"$MAMBA_BASE/envs/$ENV_NAME/bin/python" -c "import esa_climate_toolbox, xcube; print('toolbox', esa_climate_toolbox.__version__, '| xcube', xcube.__version__)"
