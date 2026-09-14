#!/usr/bin/env python3
"""Launch the SA3 music-to-music worker independently of the Colab CLI socket."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


REMOTE_COLAB_DIR = Path("/content/sa3/colab")
OUTPUT_ROOT = Path(os.environ.get("SA3_M2M_OUTPUT_ROOT", "/content/outputs/2nndPass"))
RUN_MUSIC_TO_MUSIC = REMOTE_COLAB_DIR / "run_music_to_music.py"
LAUNCH_MUSIC_TO_MUSIC = REMOTE_COLAB_DIR / "launch_music_to_music.py"
CONSOLE_LOG = OUTPUT_ROOT / "music_to_music_console.log"
EXIT_REPORT = OUTPUT_ROOT / "music_to_music_exit.json"
WORKER_REPORT = OUTPUT_ROOT / "music_to_music_worker.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run_worker() -> int:
    started_at = utc_now()
    return_code = 1
    try:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with CONSOLE_LOG.open("ab", buffering=0) as console:
            completed = subprocess.run(
                [sys.executable, str(RUN_MUSIC_TO_MUSIC)],
                cwd="/content",
                stdin=subprocess.DEVNULL,
                stdout=console,
                stderr=subprocess.STDOUT,
                check=False,
                env=os.environ.copy(),
            )
            return_code = completed.returncode
    except Exception:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with CONSOLE_LOG.open("a", encoding="utf-8") as console:
            traceback.print_exc(file=console)
    finally:
        write_json_atomic(
            EXIT_REPORT,
            {
                "started_at": started_at,
                "finished_at": utc_now(),
                "return_code": return_code,
            },
        )
    return return_code


def launch() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not RUN_MUSIC_TO_MUSIC.is_file():
        raise FileNotFoundError(f"Remote music-to-music runner not found: {RUN_MUSIC_TO_MUSIC}")
    for path in (CONSOLE_LOG, EXIT_REPORT, WORKER_REPORT):
        path.unlink(missing_ok=True)

    worker = subprocess.Popen(
        [sys.executable, str(LAUNCH_MUSIC_TO_MUSIC), "--worker"],
        cwd="/content",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
        env=os.environ.copy(),
    )
    write_json_atomic(
        WORKER_REPORT,
        {
            "launched_at": utc_now(),
            "pid": worker.pid,
            "script": str(RUN_MUSIC_TO_MUSIC),
            "console_log": str(CONSOLE_LOG),
            "exit_report": str(EXIT_REPORT),
        },
    )
    print(f"Detached Colab music-to-music worker started with PID {worker.pid}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args, _ = parser.parse_known_args()
    return run_worker() if args.worker else launch()


if __name__ == "__main__":
    raise SystemExit(main())
