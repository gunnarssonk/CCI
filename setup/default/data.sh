#!/bin/bash
export EE_PROJECT="esa-cci"
# 1. Error Handling: Stop the script if any command fails
set -e

# 2. Install uv using standard pip install uv
if ! command -v uv &> /dev/null; then
    echo -e "\e[36mInstalling uv...\e[0m"
    pip install uv
fi

# 3. Install bulk packages
uv pip install numpy matplotlib scikit-learn earthengine-api opencv-python requests tifffile rasterio tqdmuv pip install torch torchvision
echo -e "\e[36mbulk packages installed...\e[0m"

# 4. Clone and install the geotessera repository
if ! python3 -c "import geotessera" &> /dev/null; then
    echo -e "\e[36mgeotessera not found. Installing...\e[0m"
    
    # Check if we need to clone it first
    if [ ! -d "geotessera" ]; then
        git clone https://github.com/ucam-eo/geotessera
    fi

    pushd geotessera > /dev/null
    uv pip install -e .
    popd > /dev/null
else
    echo -e "\e[32mgeotessera is already installed.\e[0m"
fi

# 5. Check Earth Engine credentials
echo -e "\e[36mChecking Earth Engine credentials...\e[0m"

# Define the Python check as a heredoc
python_code=$(cat <<EOF
import ee
try:
    import os
    project = os.getenv('EE_PROJECT')
    if project:
        ee.Initialize(project=project)
    else:
        print('EE_PROJECT env var not set, using default project')
        ee.Initialize(project='alexcloud-489214')
    print('SUCCESS')
except Exception:
    print('FAIL')
EOF
)

check_result=$(uv run python3 -c "$python_code" 2>/dev/null)

if [[ "$check_result" == *"SUCCESS"* ]]; then
    echo -e "\e[32mEarth Engine already authenticated.\e[0m"
else
    echo -e "\e[33mAuthentication required. Opening browser...\e[0m"
    uv run python3 -c "import ee; ee.Authenticate()"
fi

# 6. Building the dataset
echo -e "\e[36mBuilding the dataset...\e[0m"
# Note: On Linux, we use '/' for paths and usually call the script directly
uv run python3 -m src.dataset.build