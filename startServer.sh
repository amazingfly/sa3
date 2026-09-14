#!/usr/bin/env bash
set -euo pipefail

# Get the directory of the script
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment (.venv) not found. Run ./scripts/setup_cpu.sh first."
    exit 1
fi

# Check for HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: HF_TOKEN is not set. Music generation will fail for gated models."
    echo "Please set it with: export HF_TOKEN=your_token"
fi

echo "Starting SA3 Web Server..."
# Using exec to replace the shell process with the python process
exec .venv/bin/python scripts/sa3_web_server.py --port 7860
