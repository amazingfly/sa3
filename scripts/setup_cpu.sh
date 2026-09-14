#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Creating .venv with ${PYTHON_BIN}"
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python "${PYTHON_BIN}" .venv
fi

echo "Installing CPU PyTorch and torchaudio"
uv pip install --python .venv/bin/python \
  torch==2.7.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cpu

echo "Installing Stable Audio 3 runtime dependencies"
uv pip install --python .venv/bin/python \
  "einops>=0.8.2" \
  "einops-exts>=0.0.4" \
  "numpy>=2.2.6" \
  "packaging>=26.0" \
  "safetensors>=0.7.0" \
  "tqdm>=4.67.3" \
  "huggingface-hub>=1.7.1" \
  "transformers>=5.8.0" \
  "soundfile>=0.13.1" \
  "psutil>=7.0.0" \
  "pytorch_lightning==2.5.5" \
  "dill>=0.4.1" \
  "matplotlib>=3.10.8" \
  "wandb>=0.27.0" \
  "pychromecast>=14.0.10"

echo "Installing official Stable Audio 3 package without dependency resolution"
uv pip install --python .venv/bin/python --no-deps \
  "git+https://github.com/Stability-AI/stable-audio-3.git@fa5ee841dd49bae0fa361fac26904adc27fd400e"

echo "Setup complete. Run:"
echo ".venv/bin/python scripts/run_sa3_workflow.py doctor --config configs/cyberpunk_industrial.json"
