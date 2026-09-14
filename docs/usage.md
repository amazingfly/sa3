# Stable Audio 3 Small CPU Workflow

This project runs Stable Audio 3 Small Music on CPU with prompts and generation settings controlled by JSON.

The target model is gated on Hugging Face. Before generation works, accept the model terms for `stabilityai/stable-audio-3-small-music` and make a token available:

```bash
export HF_TOKEN=hf_your_token_here
```

## Setup

```bash
./scripts/setup_cpu.sh
```

This installs CPU PyTorch first, then installs the official Stability AI `stable-audio-3` package without pulling CUDA wheels.

## Run

Validate config and environment:

```bash
.venv/bin/python scripts/run_sa3_workflow.py doctor --config configs/cyberpunk_industrial.json
```

Run the full workflow:

```bash
.venv/bin/python scripts/run_sa3_workflow.py all --config configs/cyberpunk_industrial.json
```

If the post-trained `small-music` checkpoint is not authorized for the saved Hugging Face account, use the base fallback to validate CPU/RAM behavior:

```bash
.venv/bin/python scripts/run_sa3_workflow.py all --config configs/cyberpunk_industrial_base_fallback.json
```

The base checkpoint is intended for fine-tuning, so its direct generations may be rougher than the post-trained music model.

## Web UI

The project includes a web interface for easy music generation and library management.

- `Small Music` keeps using the existing local generation workflow.
- `Medium` runs through the Google Colab CLI on a GPU and produces FLAC only.
- Medium generation accepts the normal web form, a legacy web JSON config with
  `tracks`, or a native Colab JSON config with `generation.prompt`.

### Start Web Server

```bash
./startServer.sh
```

Or manually:

```bash
export HF_TOKEN=hf_your_token_here
.venv/bin/python scripts/sa3_web_server.py --port 7860
```

The server will listen on all interfaces (0.0.0.0) so you can access it from other devices on your LAN.

For Medium, select `Medium - Colab GPU direct FLAC`. Its default and maximum
duration are 380 seconds. Submit the job and keep the automatically opened
Colab browser tab open until generation completes. The web server creates a
unique Colab session, normalizes the submitted config for the Medium model,
downloads the generated FLAC, and stops the runtime.

The Advanced `Output Name` is used at the start of the downloaded FLAC
filename. A unique web job ID is appended so generating the same name again
does not overwrite an earlier track.

Medium uses a queue to amortize Colab startup and model loading:

1. Select Medium and enter the first prompt and settings.
2. Click `Queue`.
3. Repeat for additional tracks.
4. Click `Generate Queue`.

The browser preserves the pending queue locally. The server submits the complete
list as one Colab job, installs dependencies once, loads the Medium model once,
generates every item sequentially, downloads all FLAC files, and stops the
runtime. After download, each FLAC is linked into `musicLibrary/flac/` and
converted locally with FFmpeg to Vorbis OGG quality 6 under
`musicLibrary/ogg/`. Small Music retains its existing immediate `Generate`
behavior.

Medium job files are written under
`outputs/<timestamp>-web-medium-colab-<job-id>/`. The generated FLAC is also
linked into `musicLibrary/flac/` so it is immediately playable and downloadable
from the web library. Colab setup and troubleshooting details are in
[colab/README.md](../colab/README.md).

## LoRA Training

Stable Audio 3 training in this repo is LoRA fine-tuning, not full pretraining. The base model weights stay frozen and the training produces a small adapter checkpoint that can be loaded during generation.

Prepare the MP3s in `source/` as audio/caption pairs:

```bash
.venv/bin/python scripts/prepare_lora_dataset.py --config configs/cybermetal_lora_small_cpu.json
```

Validate the local training setup:

```bash
.venv/bin/python scripts/train_sa3_lora.py doctor --config configs/cybermetal_lora_small_cpu.json
```

Run a one-step CPU smoke test:

```bash
.venv/bin/python scripts/train_sa3_lora.py smoke --config configs/cybermetal_lora_small_cpu.json
```

Run the longer configured LoRA job:

```bash
.venv/bin/python scripts/train_sa3_lora.py train --config configs/cybermetal_lora_small_cpu.json
```

The current config is conservative for 16GB CPU-only hardware. Official Stable Audio 3 LoRA docs recommend a CUDA GPU and at least 20-50 captioned clips; this local dataset currently has 10 source tracks, so expect a rough style adapter rather than a robust model.

Generate with the pilot LoRA checkpoint:

```bash
.venv/bin/python scripts/run_sa3_workflow.py full --config configs/cybermetal_lora_pilot_generate.json
```

Run the full ACE-derived prompt set with the pilot LoRA:

```bash
.venv/bin/python scripts/run_sa3_workflow.py full --config configs/ace_cybermetal_sa3_lora_pilot_full.json
```

The `all` command runs:

1. a short smoke generation,
2. repeated RAM-monitored duration probes,
3. high-setting prompt variations at lengths that survived the probe.

Outputs are written under `outputs/<timestamp>/` with `manifest.json`, per-job logs, and generated WAV files.

## Config

Edit [configs/cyberpunk_industrial.json](../configs/cyberpunk_industrial.json) to change prompts, durations, steps, CFG scale, seeds, CPU thread count, and memory guardrails.

For Stable Audio 3 Small Music, the documented maximum duration is 120 seconds. Longer values are rejected by this workflow for `small-music`.
