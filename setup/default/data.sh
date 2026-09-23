#!/bin/bash
# Script sets up dependencies, then builds the dataset via src/dataset/build_gee.py.

# Stop script if any command fails
set -e

# Figure out the absolute path of the folder this script lives in
# (so "source" works no matter where you call data.sh from).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pull in the shared functions/setup from shared_setup.sh:
# this installs uv, defines install_geotessera() and check_ee_auth(),
# and exports EE_PROJECT.
source "$SCRIPT_DIR/shared_setup.sh"

# Install the Python packages this dataset-building step needs.
uv pip install numpy matplotlib scikit-learn earthengine-api opencv-python requests tifffile rasterio tqdm
uv pip install torch torchvision

# -e enables ANSI color codes (cyan text here)
echo -e "\e[36mbulk packages installed...\e[0m"

# Run the two functions defined in shared_setup.sh:
install_geotessera
check_ee_auth

# Build the dataset by running the Python module
# Note: On Linux, we use '/' for paths and usually call the script directly
# src/dataset/build_gee.py as a module ("-m src.dataset.build_gee").

echo -e "\e[36mBuilding the dataset...\e[0m"

uv run python3 -m src.dataset.build_gee
