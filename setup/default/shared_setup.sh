#!/usr/bin/env bash
# Shared setup logic for data.sh and train.sh:
# uv install, geotessera install, and Earth Engine auth check.

# Stop if any command fails.
set -e

# Set an environment variable other commands/scripts can read.
export EE_PROJECT="esa-cci"

# Check if the "uv" command exists on this machine; install it if not.
if ! command -v uv &> /dev/null; then
    echo -e "\e[36mInstalling uv...\e[0m"
    pip install uv
fi

# Function does nothing by here — data.sh/train.sh call install_geotessera later on to run this code.
install_geotessera() {
    # Only install geotessera if it can't already be imported in Python.
    if ! python3 -c "import geotessera" &> /dev/null; then
        echo -e "\e[36mgeotessera not found. Installing...\e[0m"

        # Download the geotessera source code if we don't have it yet.
        if [ ! -d "geotessera" ]; then
            git clone https://github.com/ucam-eo/geotessera
        fi

        # Temporarily move into the geotessera folder, install it, then move back.
        pushd geotessera > /dev/null
        uv pip install -e .
        popd > /dev/null
    else
        echo -e "\e[32mgeotessera is already installed.\e[0m"
    fi
}

# Function checks (and if needed, sets up) Google Earth Engine login.
check_ee_auth() {
    echo -e "\e[36mChecking Earth Engine credentials...\e[0m"

    # It tries to connect to Earth Engine and prints SUCCESS or FAIL.
    # --- everything below, until the closing EOF, is Python code, not bash ---
    python_code=$(cat <<EOF
import ee
import os
try:
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
    # --- end of Python code ---

    # Run that Python script. 2>/dev/null hides any error text so only
    # the SUCCESS/FAIL print gets captured into check_result.
    check_result=$(uv run python3 -c "$python_code" 2>/dev/null)

    # * means "any text" — this checks if the word SUCCESS appears anywhere in check_result.
    if [[ "$check_result" == *"SUCCESS"* ]]; then
        echo -e "\e[32mEarth Engine already authenticated.\e[0m"
    else
        echo -e "\e[33mAuthentication required. Opening browser...\e[0m"
        uv run python3 -c "import ee; ee.Authenticate()"
    fi
}
