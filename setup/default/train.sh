#!/usr/bin/env bash
# Sets up dependencies, then trains the model via src/train.py.

# Stop if any command fails.
set -e

# Figure out the absolute path of the folder this script lives in
# (so "source" below works no matter where you call train.sh from).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pull in the shared functions/setup from shared_setup.sh:
# this installs uv, defines install_geotessera() and check_ee_auth(),
# and exports EE_PROJECT.
source "$SCRIPT_DIR/shared_setup.sh"

# Install the Python packages this training step needs.
uv pip install numpy matplotlib scikit-learn earthengine-api opencv-python tqdm wandb
uv pip install torch torchvision

echo -e "\e[36mbulk packages installed...\e[0m"

# Run the two functions defined in shared_setup.sh:
install_geotessera
check_ee_auth

# ==============================
# Run training
# ==============================
echo "Starting training..."

# Tells Python to also look in the current folder (".") when importing code,
# so "import src..." works correctly.
export PYTHONPATH="."

# The actual payload: runs src/train.py as a module, which trains the model.

uv run python -m src.train
