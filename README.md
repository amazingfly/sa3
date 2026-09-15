# SA3 audio workflows

Generate and review music with Stable Audio 3: local CPU generation, Colab Medium
queues, continuation, LoRA training, and a browser-based music library.

## Setup and run

Use Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/), and FFmpeg/ffprobe.
The CPU installer pins PyTorch and the upstream Stable Audio commit, then installs
its runtime dependencies. Gated models require access and `HF_TOKEN` locally.

```bash
./scripts/setup_cpu.sh
.venv/bin/python scripts/run_sa3_workflow.py doctor --config configs/cyberpunk_industrial.json
./startServer.sh
```

The UI defaults to port 7860 on all interfaces. To bind only to localhost:

```bash
.venv/bin/python scripts/sa3_web_server.py --host 127.0.0.1 --port 7860
```

See [workflow instructions](docs/usage.md) and [Colab setup](colab/README.md)
for generation, queueing, model access, and training. Medium jobs use an
independently authenticated Colab CLI; CPU setup alone does not provision Colab.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `scripts/` | Local generation, audio preparation, queue adapters, and web server |
| `colab/` | Remote generation, music-to-music, and session setup |
| `configs/` | Generation/training presets and prompt collections |
| `web/`, `static/` | Browser interface and assets |
| `tests/` | CPU-only configuration and queue contract checks |
| `archive/` | Historical UI drafts and superseded server copies |
| `docs/` | Operating instructions and storage conventions |

`outputs`, `musicLibrary`, `web_runs`, `source`, `training_data`, and `lora_runs`
are local data, excluded from Git. Existing storage symlinks are supported.
The optional image tab calls the separate `images` checkout using `SD15_REPO`;
video assembly and cross-repository orchestration live in `ltx-video` and
`media-pipeline` respectively.

## Checks without model downloads

```bash
uv venv .venv-check
uv pip install --python .venv-check/bin/python -r requirements-dev.txt
.venv-check/bin/python -m unittest discover -s tests -v
```

Audio acquisition is available in `scripts/download_audio.py`; install
`requirements-download.txt` separately if using it. It writes downloads to the
current directory, so run it from your local `source/` directory.

See [workflow support status](docs/workflows.md). Use `python scripts/workspace.py doctor`
to check centralized checkout/interpreter configuration, and
`python scripts/workspace.py run --component sa3 -- {python} SCRIPT [ARGS]`
to launch with shared paths. Workspace setup is documented in
[media-pipeline](https://github.com/amazingfly/media-pipeline/blob/main/docs/workspace.md).
