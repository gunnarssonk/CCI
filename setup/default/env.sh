#!/usr/bin/env bash
# Temporarily adds Mamba to PATH for this terminal session, then creates/activates the esa_env environment.

# 1. Define Paths
MAMBA_BASE="$HOME/miniforge3"
MAMBA_BIN="$MAMBA_BASE/bin"
echo "Detected Mamba base: $MAMBA_BASE"

ENV_NAME="esa_env"
PYTHON_VERSION="3.11"
ENV_PATH="$MAMBA_BASE/envs/$ENV_NAME"

# 2. Add to PATH for this session
if [ -d "$MAMBA_BIN" ]; then
    export PATH="$MAMBA_BIN:$PATH"
    echo -e "\e[36mMamba path linked.\e[0m"
else
    echo -e "\e[31mMamba not found at $MAMBA_BIN. Please check your installation.\e[0m"
    return 1 2>/dev/null || exit 1
fi

# 3. Initialize Shell Hook
eval "$(mamba shell hook --shell bash)"

# 4. Check if the PHYSICAL folder exists
if [ -d "$ENV_PATH" ]; then
    echo -e "\e[32mEnvironment '$ENV_NAME' already exists at $ENV_PATH. Activating...\e[0m"
else
    echo -e "\e[33mEnvironment '$ENV_NAME' not found. Creating with Python $PYTHON_VERSION...\e[0m"
    # This command creates the env AND the physical directory
    mamba create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# 5. Activate
mamba activate "$ENV_NAME"

echo "Active Environment: $ENV_NAME (Python $PYTHON_VERSION)"