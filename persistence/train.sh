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

# Train the model and save weights to model.json
# Pass through any extra args (e.g. --steps 1000 --output my_model.json)
uv run "$SCRIPT_DIR/train.py" "$@"
