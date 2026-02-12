#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install uv if not available
if ! command -v uv &>/dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv installation failed" >&2
        exit 1
    fi
    echo "uv installed successfully: $(uv --version)"
fi

# Run inference using saved model weights
# Pass through any extra args (e.g. --model my_model.json --samples 50 --temperature 0.8)
uv run "$SCRIPT_DIR/run.py" "$@"
