#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

# --- Constants & Paths ---
ROOT = Path(__file__).resolve().parents[1]
WEB_RUNS = ROOT / "web_runs"
CONFIG_DIR = WEB_RUNS / "configs"
LOG_DIR = WEB_RUNS / "logs"
OGG_DIR = ROOT / "musicLibrary" / "ogg"
FLAC_DIR = ROOT / "musicLibrary" / "flac"
WEB_DEFAULTS_PATH = ROOT / "configs" / "web_defaults.json"
SD15_REPO = Path(os.environ.get("SD15_REPO", "/mnt/storage/projects/agentic/images")).resolve()
SD15_CONFIG_PATH = SD15_REPO / "config.json"
SD15_SCRIPTS = SD15_REPO / "scripts"
SD15_OUTPUT_DIR = SD15_SCRIPTS / "output"
SD15_UPSCALED_DIR = SD15_OUTPUT_DIR / "upscaled"
SD15_PROMPTS_PATH = SD15_SCRIPTS / "prompts.json"
IMAGES_DIR = SD15_UPSCALED_DIR
IMAGES_LOG_PATH = ROOT / "images_log.json"

# --- Global State ---
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
SD_JOBS: dict[str, dict] = {}
SD_JOBS_LOCK = threading.Lock()
CAST_STATE: dict[str, object] = {
    "status": "idle",
    "device": "speaker group alpha",
    "current": None,
    "index": 0,
    "count": 0,
    "error": None,
    "started_at": None,
}
CAST_LOCK = threading.Lock()
CAST_STOP = threading.Event()
CAST_SESSION: dict[str, object] = {"cast": None, "media_controller": None}

# --- Utilities ---

def load_web_defaults() -> dict:
    if WEB_DEFAULTS_PATH.exists():
        try:
            return json.loads(WEB_DEFAULTS_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Error loading {WEB_DEFAULTS_PATH}: {exc}")
    # Fallback
    return {
        "mode": "continuation",
        "output_dir": "outputs",
        "music_library_dir": "musicLibrary",
        "model": {
            "name": "small-music",
            "hf_repo": "cocktailpeanut/stable-audio-3-small-music",
            "conditioning_repo_id_override": "cocktailpeanut/stable-audio-3-small-music",
            "device": "cpu",
            "model_half": False,
            "chunked_decode": True,
            "max_duration_seconds": 120.0,
        },
        "cpu": {"torch_num_threads": 6, "torch_num_interop_threads": 1},
        "memory": {
            "sample_interval_seconds": 0.5,
            "min_available_ram_gb": 1.5,
            "max_process_rss_gb": 13.5,
        },
        "generation": {
            "duration_seconds": 120.0,
            "steps": 30,
            "cfg_scale": 1.0,
            "apg_scale": 1.0,
            "duration_padding_sec": 6.0,
            "sampler_type": "pingpong",
            "init_noise_level": 0.76,
            "seed_start": None,
            "negative_prompt": "poor quality, harsh static, loud buzzing, clipping, noise, vocals, singing, spoken words, narrator, acoustic folk, soft ballad",
        },
        "acts": [{"id": "beginning"}, {"id": "middle"}, {"id": "end"}],
        "standard_prompts": {
            "base_prefix": "TrackType: Music, VocalType: Instrumental. This is part of a coherent 360-second cybermetal electronica suite.",
            "beginning": "Act role: beginning 120 seconds. Establish the main tempo, key, drum kit, synth palette, bass tone, and cybermetal identity. Build from intro into the first full-power groove, with 80s science-fiction synths, robots, neon circuitry, nuclear laser effects, and no vocals.",
            "middle": "Act role: middle 120 seconds, generated as a direct continuation of the same track. Preserve the previous section's tempo, key, drum kit, bass tone, synth patches, stereo image, and mastering. Escalate the arrangement with more high-speed synth arpeggios, robot-machine percussion, nuclear laser lead lines, harder drops, and denser 80s sci-fi electronica energy. Avoid abrupt style changes and keep it instrumental.",
            "end": "Act role: final 120 seconds, generated as a direct continuation of the same track. Preserve the previous section's tempo, key, drum kit, bass tone, synth patches, stereo image, and mastering. Deliver the climactic finale with blazing laser synth hooks, robot uprising energy, reactor-core risers, cybermetal stabs, a final overdrive drop, and a resolved ending. Avoid abrupt style changes and keep it instrumental.",
        },
        "tracks": [
            {
                "id": "web-cybermetal-suite",
                "prompt": "Cybermetal electronica, high-speed synth arpeggios, 80s sci-fi robots, nuclear laser leads, industrial drums",
            }
        ],
    }

def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "web-suite"

def clamp_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))

def clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))

def config_model_name(config: dict) -> str:
    model = config.get("model")
    if not isinstance(model, dict):
        return ""
    return str(model.get("name") or "")


def stored_track_id(config: dict, fallback: str) -> str:
    if config.get("track_id"):
        return slugify(str(config["track_id"]))
    tracks = config.get("tracks")
    if isinstance(tracks, list) and tracks and isinstance(tracks[0], dict):
        if tracks[0].get("id"):
            return slugify(str(tracks[0]["id"]))
    generation = config.get("generation")
    if isinstance(generation, dict) and generation.get("output_name"):
        return slugify(Path(str(generation["output_name"])).stem)
    return fallback


def build_config(job_id: str, payload: dict) -> tuple[Path, str]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if "raw_config" in payload:
        config = payload["raw_config"]
        if not isinstance(config, dict):
            raise ValueError("Raw config must be a dictionary.")
        model_name = config_model_name(config)
        tracks = config.get("tracks")
        generation = config.get("generation")
        is_native_medium = (
            model_name == "medium"
            and isinstance(generation, dict)
            and bool(str(generation.get("prompt") or "").strip())
        )
        if not is_native_medium and not isinstance(tracks, list):
            raise ValueError(
                "Raw config must use legacy tracks or native Medium generation.prompt."
            )
        if is_native_medium:
            track_id = slugify(
                str(
                    config.get("track_id")
                    or Path(str(generation.get("output_name") or "")).stem
                    or f"raw-{job_id}"
                )
            )
        else:
            first_track = tracks[0] if tracks else {}
            track_id = slugify(
                str(
                    first_track.get("id")
                    if isinstance(first_track, dict)
                    else f"raw-{job_id}"
                )
            )
        path = CONFIG_DIR / f"{job_id}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return path, track_id

    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Prompt is required.")

    defaults = load_web_defaults()
    track_id = slugify(str(payload.get("track_id") or prompt[:48]))
    requested_model = str(payload.get("model") or defaults["model"]["name"])

    if requested_model == "medium":
        model_name = "medium"
        hf_repo = "cocktailpeanut/stable-audio-3-medium"
        max_duration = 380.0
        default_duration = 380.0
    else:
        model_name = "small-music"
        hf_repo = "cocktailpeanut/stable-audio-3-small-music"
        max_duration = 120.0
        default_duration = 120.0

    continuation = bool(payload.get("continuation") if "continuation" in payload else (defaults["mode"] == "continuation"))
    if model_name == "medium":
        continuation = False
    if continuation:
        max_duration = min(max_duration, 120.0)
        default_duration = min(default_duration, 120.0)
    mode = "continuation" if continuation else "direct"

    duration = clamp_float(payload.get("duration_seconds"), default_duration, 10.0, max_duration)
    steps = clamp_int(payload.get("steps"), defaults["generation"]["steps"], 1, 80)
    cfg_scale = clamp_float(payload.get("cfg_scale"), defaults["generation"]["cfg_scale"], 0.0, 10.0)
    init_noise = clamp_float(payload.get("init_noise_level"), defaults["generation"]["init_noise_level"], 0.0, 1.0)
    seed = payload.get("seed_start")
    if seed is None or seed == "":
        seed = int(time.time()) % 900000 + 100000
    seed = clamp_int(seed, 370000, 1, 999999999)

    config = {
        "mode": mode,
        "output_dir": defaults.get("output_dir", "outputs"),
        "music_library_dir": defaults.get("music_library_dir", "musicLibrary"),
        "model": {
            "name": model_name,
            "hf_repo": hf_repo,
            "conditioning_repo_id_override": hf_repo,
            "device": "cuda" if model_name == "medium" else "cpu",
            "model_half": model_name == "medium",
            "low_memory_load": model_name == "medium",
            "chunked_decode": True,
            "max_duration_seconds": max_duration,
        },
        "cpu": defaults.get("cpu", {"torch_num_threads": 6, "torch_num_interop_threads": 1}),
        "memory": defaults.get("memory", {
            "sample_interval_seconds": 0.5,
            "min_available_ram_gb": 1.5,
            "max_process_rss_gb": 13.5,
        }),
        "generation": {
            "duration_seconds": duration,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "apg_scale": 1.0,
            "duration_padding_sec": 6.0,
            "sampler_type": "pingpong",
            "init_noise_level": init_noise,
            "seed_start": seed,
            "negative_prompt": defaults["generation"].get("negative_prompt"),
        },
        "acts": defaults.get("acts", [{"id": "beginning"}, {"id": "middle"}, {"id": "end"}]),
        "standard_prompts": defaults.get("standard_prompts", {}),
        "tracks": [
            {
                "id": track_id,
                "prompt": prompt,
            }
        ],
    }
    path = CONFIG_DIR / f"{job_id}.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path, track_id


def build_queue_config(job_id: str, payload: dict) -> tuple[Path, list[str]]:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Queue must contain at least one item.")
    if len(items) > 20:
        raise ValueError("Medium queue supports at most 20 items.")

    configs = []
    track_ids = []
    seed_base = int(time.time() * 1000) % 900000000
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Queue item {index} must be an object.")
        raw_config = item.get("raw_config")
        if isinstance(raw_config, dict):
            submitted_generation = raw_config.get("generation", {})
            submitted_seed = (
                submitted_generation.get("seed")
                if isinstance(submitted_generation, dict)
                else None
            )
            if submitted_seed in (None, "") and isinstance(
                submitted_generation, dict
            ):
                submitted_seed = submitted_generation.get("seed_start")
        else:
            submitted_seed = item.get("seed_start")
        item_path, track_id = build_config(f"{job_id}-{index:03d}", item)
        try:
            config = json.loads(item_path.read_text(encoding="utf-8"))
        finally:
            item_path.unlink(missing_ok=True)
        if config_model_name(config) != "medium":
            raise ValueError(
                f"Queue item {index} is not an SA3 Medium configuration."
            )
        config["track_id"] = track_id
        generation = config.get("generation")
        if not isinstance(generation, dict):
            raise ValueError(f"Queue item {index} has no generation object.")
        if submitted_seed in (None, ""):
            generation.pop("seed", None)
            generation["seed_start"] = seed_base + index
        configs.append(config)
        track_ids.append(track_id)

    queue_config = {
        "queue_version": 1,
        "model": {"name": "medium"},
        "items": configs,
    }
    path = CONFIG_DIR / f"{job_id}.json"
    path.write_text(json.dumps(queue_config, indent=2), encoding="utf-8")
    return path, track_ids

def log_tail(log_path: str | None, max_chars: int = 6000) -> str:
    if not log_path:
        return ""
    path = Path(log_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def generation_failure_message(log_path: Path, return_code: int) -> str:
    tail = log_tail(str(log_path), max_chars=20000)
    if "Colab runtime disappeared during queue generation." in tail:
        return (
            "The Colab runtime disappeared during generation before the "
            "audio output was returned."
        )
    if "CUDA out of memory" in tail:
        return "Colab ran out of GPU memory during generation."
    if "Generation failed:" in tail:
        message = tail.rsplit("Generation failed:", 1)[-1].strip().splitlines()
        if message:
            return f"Generation failed: {message[0]}"
    return f"Generator exited with code {return_code}."


def hf_home_for_jobs() -> str:
    configured = os.environ.get("SA3_HF_HOME")
    if configured:
        return configured
    storage = Path("/mnt/storage")
    if storage.exists() and os.access(storage, os.W_OK):
        return str(storage / "sa3_hf_home")
    return str(ROOT / ".hf_home")

def refresh_job_outputs(job: dict) -> None:
    manifest_path = job.get("manifest_path")
    if not manifest_path:
        run_dir = job.get("run_dir")
        if run_dir:
            candidate = Path(run_dir) / "manifest.json"
            if candidate.exists():
                manifest_path = str(candidate)
                job["manifest_path"] = manifest_path
    if not manifest_path or not Path(manifest_path).exists():
        return
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        outputs = []
        for track in manifest.get("tracks", []):
            combined = track.get("combined", {})
            ogg_path = combined.get("library_ogg")
            output_flac = combined.get("output_flac")
            flac_path = combined.get("library_flac") or output_flac
            if ogg_path and Path(ogg_path).exists():
                audio_path = ogg_path
                audio_format = "ogg"
                download_name = Path(ogg_path).name
            elif flac_path and Path(flac_path).exists():
                audio_path = flac_path
                audio_format = "flac"
                download_name = Path(output_flac or flac_path).name
            else:
                continue
            outputs.append(
                {
                    "track_id": track.get("id"),
                    "audio_path": audio_path,
                    "audio_format": audio_format,
                    "download_name": download_name,
                    "audio_url": (
                        f"/audio/{job['id']}/{quote(download_name)}"
                    ),
                }
            )
        job["outputs"] = outputs
        if outputs:
            first = outputs[0]
            job["audio_path"] = first["audio_path"]
            job["audio_format"] = first["audio_format"]
            job["download_name"] = first["download_name"]
            job["audio_url"] = first["audio_url"]
            if first["audio_format"] == "flac":
                job["flac_path"] = first["audio_path"]
            else:
                job["ogg_path"] = first["audio_path"]
        if outputs and len(outputs) == int(job.get("queue_count") or 1):
            job["status"] = "done"
            job["error"] = None
    except Exception as exc:
        job["error"] = f"Could not parse manifest: {exc}"


def load_existing_jobs() -> None:
    """Restore completed and failed SA3 jobs after a web-server restart."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output_root = ROOT / "outputs"
    for config_path in sorted(CONFIG_DIR.glob("*.json")):
        job_id = config_path.stem
        if job_id.startswith("sd-"):
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            items = config.get("items")
            configs = items if isinstance(items, list) else [config]
            track_ids = [
                stored_track_id(item, f"track-{index}")
                for index, item in enumerate(configs, start=1)
            ]
            model_name = config_model_name(config)
        except Exception:
            continue

        log_path = LOG_DIR / f"{job_id}.log"
        run_dirs = []
        if output_root.is_dir():
            run_dirs = sorted(
                (
                    path
                    for path in output_root.iterdir()
                    if path.is_dir() and path.name.endswith(f"-{job_id}")
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        completed_run = next(
            (path for path in run_dirs if (path / "manifest.json").is_file()),
            None,
        )
        queue_count = len(track_ids)
        job = {
            "id": job_id,
            "track_id": (
                f"Medium queue ({queue_count} tracks)"
                if queue_count > 1 and model_name == "medium"
                else track_ids[0]
            ),
            "status": "done" if completed_run else "failed",
            "created_at": datetime.fromtimestamp(
                config_path.stat().st_mtime
            ).isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "queue_count": queue_count,
            "backend": "colab-cli" if model_name == "medium" else "local",
            "log_path": str(log_path),
        }
        if completed_run:
            job["run_dir"] = str(completed_run)
            job["manifest_path"] = str(completed_run / "manifest.json")
            refresh_job_outputs(job)
        elif log_path.is_file():
            job["error"] = generation_failure_message(log_path, 1)
        JOBS[job_id] = job


def run_job(job_id: str, config_path: Path) -> None:
    log_path = LOG_DIR / f"{job_id}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_name = config_model_name(config)
    if model_name == "medium":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_sa3_medium_colab.py"),
            "--config",
            str(config_path),
            "--job-id",
            job_id,
        ]
        backend = "colab-cli"
    else:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_sa3_web_generation.py"),
            "--config",
            str(config_path),
        ]
        backend = "local"
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = "running"
        job["log_path"] = str(log_path)
        job["backend"] = backend
    run_dir = None
    with log_path.open("w", encoding="utf-8") as log:
        env = os.environ.copy()
        env["HF_HOME"] = hf_home_for_jobs()
        if not env.get("HF_TOKEN"):
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            if token_path.exists():
                env["HF_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
        Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
        log.write(f"HF_HOME={env['HF_HOME']}\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        with JOBS_LOCK:
            JOBS[job_id]["pid"] = process.pid
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith("Run directory:"):
                run_dir = line.split(":", 1)[1].strip()
                with JOBS_LOCK:
                    JOBS[job_id]["run_dir"] = run_dir
                    JOBS[job_id]["manifest_path"] = str(Path(run_dir) / "manifest.json")
        rc = process.wait()
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["return_code"] = rc
        if rc == 0:
            job["status"] = "done"
            refresh_job_outputs(job)
            if not job.get("outputs"):
                job["status"] = "failed"
                job["error"] = "Generation finished but no playable audio output was found."
        else:
            job["status"] = "failed"
            job["error"] = generation_failure_message(log_path, rc)

def any_running() -> bool:
    return any(job.get("status") in {"queued", "running"} for job in JOBS.values())

def safe_ogg_path(filename: str) -> Path | None:
    if not filename or Path(filename).name != filename:
        return None
    path = (OGG_DIR / filename).resolve()
    try:
        path.relative_to(OGG_DIR.resolve())
    except ValueError:
        return None
    if not path.exists() or path.suffix.lower() != ".ogg":
        return None
    return path

def list_ogg_tracks() -> list[dict]:
    OGG_DIR.mkdir(parents=True, exist_ok=True)
    tracks = []
    for path in sorted(OGG_DIR.glob("*.ogg"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        tracks.append(
            {
                "filename": path.name,
                "title": path.stem,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "url": f"/library/audio/{quote(path.name)}",
            }
        )
    return tracks

def safe_library_audio_path(filename: str) -> Path | None:
    if not filename or Path(filename).name != filename:
        return None
    suffix = Path(filename).suffix.lower()
    base_dir = OGG_DIR if suffix == ".ogg" else FLAC_DIR if suffix == ".flac" else None
    if base_dir is None:
        return None
    path = (base_dir / filename).resolve()
    try:
        path.relative_to(base_dir.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None

def list_audio_tracks() -> list[dict]:
    tracks = []
    for base_dir, suffix, audio_format in (
        (OGG_DIR, "*.ogg", "ogg"),
        (FLAC_DIR, "*.flac", "flac"),
    ):
        base_dir.mkdir(parents=True, exist_ok=True)
        for path in base_dir.glob(suffix):
            stat = path.stat()
            tracks.append(
                {
                    "filename": path.name,
                    "title": path.stem,
                    "format": audio_format,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(
                        timespec="seconds"
                    ),
                    "url": f"/library/audio/{quote(path.name)}",
                }
            )
    tracks.sort(key=lambda item: item["modified"], reverse=True)
    return tracks

def cast_devices(timeout: float = 5.0) -> dict:
    try:
        import pychromecast
    except Exception as exc:
        return {"available": False, "error": f"pychromecast is not available: {exc}", "devices": []}
    browser = None
    try:
        chromecasts, browser = pychromecast.get_chromecasts(timeout=timeout)
        devices = []
        for cast in chromecasts:
            info = getattr(cast, "cast_info", None)
            name = getattr(info, "friendly_name", None) or getattr(cast, "name", "")
            devices.append(
                {
                    "name": name,
                    "uuid": str(getattr(info, "uuid", "")),
                    "model": getattr(info, "model_name", ""),
                }
            )
        devices.sort(key=lambda item: item["name"].lower())
        return {"available": True, "error": None, "devices": devices}
    except Exception as exc:
        return {"available": False, "error": str(exc), "devices": []}
    finally:
        if browser is not None:
            try:
                pychromecast.discovery.stop_discovery(browser)
            except Exception:
                pass

def cast_name_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())

def find_cast(device_name: str, timeout: float = 10.0):
    import pychromecast
    browser = None
    chromecasts, browser = pychromecast.get_chromecasts(timeout=timeout)
    target = cast_name_key(device_name)
    try:
        for cast in chromecasts:
            info = getattr(cast, "cast_info", None)
            name = getattr(info, "friendly_name", None) or getattr(cast, "name", "")
            if cast_name_key(name) == target:
                cast.wait(timeout=timeout)
                return cast, browser
        names = ", ".join(
            (getattr(getattr(cast, "cast_info", None), "friendly_name", None) or getattr(cast, "name", ""))
            for cast in chromecasts
        )
        raise RuntimeError(f"Cast target not found: {device_name}. Found: {names or 'none'}")
    except Exception:
        if browser is not None:
            try:
                pychromecast.discovery.stop_discovery(browser)
            except Exception:
                pass
        raise

def update_cast_state(**values) -> None:
    with CAST_LOCK:
        CAST_STATE.update(values)

def cast_playlist_worker(device_name: str, filenames: list[str], base_url: str) -> None:
    import pychromecast
    browser = None
    cast = None
    media_controller = None
    try:
        update_cast_state(
            status="connecting",
            device=device_name,
            current=None,
            index=0,
            count=len(filenames),
            error=None,
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        CAST_STOP.clear()
        cast, browser = find_cast(device_name)
        media_controller = cast.media_controller
        with CAST_LOCK:
            CAST_SESSION["cast"] = cast
            CAST_SESSION["media_controller"] = media_controller
        for index, filename in enumerate(filenames, start=1):
            if CAST_STOP.is_set():
                break
            path = safe_ogg_path(filename)
            if path is None:
                continue
            url = f"{base_url}/library/audio/{quote(path.name)}"
            update_cast_state(status="playing", current=path.name, index=index, count=len(filenames), error=None)
            media_controller.play_media(url, "audio/ogg", title=path.stem, stream_type="BUFFERED")
            media_controller.block_until_active(timeout=20)
            started = time.monotonic()
            saw_playing = False
            while not CAST_STOP.is_set():
                media_controller.update_status()
                player_state = str(getattr(media_controller.status, "player_state", "") or "")
                if player_state == "PLAYING":
                    saw_playing = True
                if player_state in {"IDLE", "UNKNOWN"} and (saw_playing or time.monotonic() - started > 20):
                    break
                time.sleep(2)
        if CAST_STOP.is_set():
            update_cast_state(status="stopped")
            try:
                media_controller.stop()
            except Exception:
                pass
        else:
            update_cast_state(status="idle", current=None)
    except Exception as exc:
        update_cast_state(status="error", error=str(exc))
    finally:
        with CAST_LOCK:
            CAST_SESSION["cast"] = None
            CAST_SESSION["media_controller"] = None
        if browser is not None:
            try:
                pychromecast.discovery.stop_discovery(browser)
            except Exception:
                pass

def start_cast_playlist(device_name: str, filenames: list[str], base_url: str) -> None:
    if not filenames:
        raise ValueError("No OGG tracks selected.")
    for filename in filenames:
        if safe_ogg_path(filename) is None:
            raise ValueError(f"Unknown OGG track: {filename}")
    if importlib.util.find_spec("pychromecast") is None:
        raise RuntimeError("pychromecast is not available")
    CAST_STOP.set()
    thread = threading.Thread(
        target=cast_playlist_worker,
        args=(device_name or "speaker group alpha", filenames, base_url),
        daemon=True,
    )
    thread.start()

def lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("1.1.1.1", 80))
        return sock.getsockname()[0]
    finally:
        sock.close()

def sd15_python() -> str:
    candidate = ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable

def any_sd_running() -> bool:
    return any(job.get("status") in {"queued", "running"} for job in SD_JOBS.values())

def sd15_resize_settings(value: str) -> tuple[int, int]:
    if value == "720p":
        return 1280, 720
    if value == "480p":
        return 854, 480
    return 1920, 1080

def load_sd15_config() -> dict:
    if not SD15_CONFIG_PATH.exists():
        raise FileNotFoundError(f"SD1.5 config not found: {SD15_CONFIG_PATH}")
    return json.loads(SD15_CONFIG_PATH.read_text(encoding="utf-8"))

def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def build_sd15_job_config(job_id: str, payload: dict, mode: str) -> Path:
    config = load_sd15_config()
    generation = dict(config.get("generation", {}))
    upscale = dict(config.get("upscale", {}))

    width, height = sd15_resize_settings(str(payload.get("resize") or "1080p"))
    upscale.update({"width": width, "height": height})

    steps = clamp_int(payload.get("steps"), int(generation.get("num_inference_steps", 50)), 1, 100)
    generation["num_inference_steps"] = steps
    generation["skip_existing"] = False

    config["output_dir"] = str(SD15_OUTPUT_DIR)
    config["upscaled_dir"] = str(SD15_UPSCALED_DIR)
    config["local_files_only"] = bool(config.get("local_files_only", True))
    config["generation"] = generation
    config["upscale"] = upscale

    if mode == "manual":
        prompt = str(payload.get("prompt") or "dystopian cyborg").strip()
        prompt_path = CONFIG_DIR / f"sd-{job_id}-prompts.json"
        write_json(prompt_path, [prompt])
        config["prompt_count"] = 1
        config["prompts_file"] = str(prompt_path)
    else:
        count = clamp_int(payload.get("prompt_count"), int(config.get("prompt_count", 50)), 1, 200)
        config["prompt_count"] = count
        config["prompts_file"] = str(SD15_PROMPTS_PATH)
        config.setdefault("pipeline", {})["regenerate_prompts"] = True
        config["prompt_request"] = str(payload.get("gemma_prompt") or config.get("prompt_request") or "").strip()

        persisted = load_sd15_config()
        persisted["prompt_count"] = count
        persisted.setdefault("pipeline", {})["regenerate_prompts"] = True
        persisted_generation = dict(persisted.get("generation", {}))
        persisted_generation["num_inference_steps"] = steps
        persisted_generation["skip_existing"] = False
        persisted["generation"] = persisted_generation
        persisted_upscale = dict(persisted.get("upscale", {}))
        persisted_upscale.update({"width": width, "height": height})
        persisted["upscale"] = persisted_upscale
        if config["prompt_request"]:
            persisted["prompt_request"] = config["prompt_request"]
        write_json(SD15_CONFIG_PATH, persisted)

    job_config_path = CONFIG_DIR / f"sd-{job_id}.json"
    write_json(job_config_path, config)
    return job_config_path

def build_sd15_command(config_path: Path, mode: str) -> list[str]:
    python = shlex.quote(sd15_python())
    config_arg = shlex.quote(str(config_path))
    generate_images = shlex.quote(str(SD15_SCRIPTS / "generate_images.py"))

    if mode == "manual":
        command = f"{python} -u {generate_images} {config_arg}"
    else:
        generate_prompts = shlex.quote(str(SD15_SCRIPTS / "generate_prompts.py"))
        command = (
            f"{python} -u {generate_prompts} {config_arg} && "
            "sleep 15 && "
            f"{python} -u {generate_images} {config_arg}"
        )
    return ["/bin/bash", "-lc", command]

# --- HTTP Handler ---

def run_sd_job(job_id: str, cmd_parts: list[str], mode: str) -> None:
    log_path = LOG_DIR / f"sd-{job_id}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with SD_JOBS_LOCK:
        job = SD_JOBS[job_id]
        job["status"] = "running"
        job["log_path"] = str(log_path)

    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"Starting SD Job {job_id} in {SD15_REPO}\n")
            log.write(f"Command: {' '.join(cmd_parts)}\n")
            log.flush()

            env = os.environ.copy()
            env.setdefault("MALLOC_ARENA_MAX", "2")
            process = subprocess.Popen(
                cmd_parts,
                cwd=str(SD15_REPO),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
            rc = process.wait()

        with SD_JOBS_LOCK:
            SD_JOBS[job_id]["return_code"] = rc
            if rc == 0:
                SD_JOBS[job_id]["status"] = "done"
            else:
                SD_JOBS[job_id]["status"] = "failed"
                SD_JOBS[job_id]["error"] = f"Exit code {rc}"
    except Exception as exc:
        with SD_JOBS_LOCK:
            SD_JOBS[job_id]["status"] = "failed"
            SD_JOBS[job_id]["error"] = str(exc)

class Handler(SimpleHTTPRequestHandler):
    server_version = "SA3Web/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def base_url(self) -> str:
        return f"http://{lan_ip()}:{self.server.server_address[1]}"

    def send_file_response(self, path: Path, content_type: str, filename: str, send_body: bool = True) -> None:
        total = path.stat().st_size
        start = 0
        end = total - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            raw_range = range_header.split("=", 1)[1].split(",", 1)[0].strip()
            raw_start, _, raw_end = raw_range.partition("-")
            try:
                if raw_start:
                    start = int(raw_start)
                if raw_end:
                    end = int(raw_end)
                if start < 0 or end < start or start >= total:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{total}")
                    self.end_headers()
                    return
                end = min(end, total - 1)
                status = HTTPStatus.PARTIAL_CONTENT
            except ValueError:
                start = 0
                end = total - 1

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Disposition", f'inline; filename="{html.escape(filename)}"')
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()
        if not send_body:
            return
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def send_library_audio(self, parsed, send_body: bool = True) -> bool:
        filename = unquote(parsed.path.removeprefix("/library/audio/"))
        audio_path = safe_library_audio_path(filename)
        if audio_path is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        content_type = "audio/flac" if audio_path.suffix.lower() == ".flac" else "audio/ogg"
        self.send_file_response(
            audio_path,
            content_type,
            audio_path.name,
            send_body=send_body,
        )
        return True

    def send_job_audio(self, parsed, send_body: bool = True) -> bool:
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) != 3:
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        _, job_id, filename = parts
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            output = next(
                (
                    item
                    for item in (job.get("outputs", []) if job else [])
                    if item.get("download_name") == filename
                ),
                None,
            )
            if output is None and job and job.get("download_name") == filename:
                output = {
                    "audio_path": job.get("audio_path"),
                    "audio_format": job.get("audio_format"),
                    "download_name": job.get("download_name"),
                }
        audio_path = Path(output.get("audio_path", "")) if output else None
        audio_format = output.get("audio_format") if output else None
        download_name = output.get("download_name") if output else None
        if (
            not job
            or not audio_path
            or not audio_path.exists()
            or download_name != filename
        ):
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        content_type = "audio/flac" if audio_format == "flac" else "audio/ogg"
        self.send_file_response(
            audio_path,
            content_type,
            download_name,
            send_body=send_body,
        )
        return True

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        # Rewrite routes for SimpleHTTPRequestHandler
        if parsed.path == "/":
            self.path = "/index.html"
            return super().do_GET()
        if parsed.path == "/library":
            self.path = "/library.html"
            return super().do_GET()

        # API Endpoints
        if parsed.path == "/api/defaults":
            self.send_json(load_web_defaults())
            return
        if parsed.path == "/api/jobs":
            with JOBS_LOCK:
                jobs = []
                for job in sorted(JOBS.values(), key=lambda j: j["created_at"], reverse=True):
                    refresh_job_outputs(job)
                    jobs.append(
                        {
                            "id": job["id"],
                            "track_id": job.get("track_id"),
                            "status": job.get("status"),
                            "created_at": job.get("created_at"),
                            "audio_url": job.get("audio_url"),
                            "audio_format": job.get("audio_format"),
                            "download_name": job.get("download_name"),
                            "flac_path": job.get("flac_path"),
                            "outputs": job.get("outputs", []),
                            "queue_count": job.get("queue_count"),
                            "backend": job.get("backend"),
                            "error": job.get("error"),
                            "log_tail": log_tail(job.get("log_path")),
                        }
                    )
                running = any(job["status"] in {"queued", "running"} for job in jobs)
            self.send_json({"jobs": jobs, "running": running})
            return
        if parsed.path == "/api/sd15/jobs":
            with SD_JOBS_LOCK:
                jobs = []
                for job in sorted(SD_JOBS.values(), key=lambda j: j["created_at"], reverse=True):
                    jobs.append({
                        "id": job["id"],
                        "status": job["status"],
                        "mode": job["mode"],
                        "created_at": job["created_at"],
                        "config_path": job.get("config_path"),
                        "output_dir": job.get("output_dir"),
                        "upscaled_dir": job.get("upscaled_dir"),
                        "log_tail": log_tail(job.get("log_path")),
                        "error": job.get("error")
                    })
                running = any(j["status"] in {"queued", "running"} for j in jobs)
            self.send_json({"jobs": jobs, "running": running})
            return
        if parsed.path == "/api/images":
            images = []
            if IMAGES_DIR.exists():
                files = sorted(IMAGES_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
                for path in files:
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                        images.append({"filename": path.name, "url": f"/images/{quote(path.name)}"})
            self.send_json({"images": images})
            return
        if parsed.path == "/api/library":
            self.send_json({"tracks": list_audio_tracks()})
            return
        if parsed.path == "/api/cast/devices":
            self.send_json(cast_devices())
            return
        if parsed.path == "/api/cast/status":
            with CAST_LOCK:
                self.send_json(dict(CAST_STATE))
            return

        # File Serving Overrides
        if parsed.path.startswith("/images/"):
            filename = unquote(parsed.path.removeprefix("/images/"))
            path = (IMAGES_DIR / filename).resolve()
            if path.parent == IMAGES_DIR.resolve() and path.exists():
                self.send_file_response(path, "image/png", path.name)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
            return
        if parsed.path.startswith("/library/audio/"):
            self.send_library_audio(parsed)
            return
        if parsed.path.startswith("/audio/"):
            self.send_job_audio(parsed)
            return

        # Fallback to SimpleHTTPRequestHandler
        super().do_GET()

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/library/audio/"):
            self.send_library_audio(parsed, send_body=False)
            return
        if parsed.path.startswith("/audio/"):
            self.send_job_audio(parsed, send_body=False)
            return
        super().do_HEAD()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw_body)
        except Exception:
            self.send_json({"error": "Invalid JSON."}, status=400)
            return

        if parsed.path == "/api/cast/play":
            device_name = str(payload.get("device_name") or "speaker group alpha")
            playlist = payload.get("playlist")
            if not isinstance(playlist, list):
                playlist = [track["filename"] for track in list_ogg_tracks()]
            filenames = [str(item) for item in playlist]
            try:
                start_cast_playlist(device_name, filenames, self.base_url())
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)
                return
            self.send_json(
                {
                    "message": f"Started cast playlist on {device_name}: {len(filenames)} tracks.",
                    "device": device_name,
                    "count": len(filenames),
                }
            )
            return
        if parsed.path == "/api/cast/stop":
            CAST_STOP.set()
            with CAST_LOCK:
                controller = CAST_SESSION.get("media_controller")
            if controller is not None:
                try:
                    controller.stop()
                except Exception:
                    pass
            update_cast_state(status="stopped", current=None)
            self.send_json({"message": "Cast stop requested."})
            return
        if parsed.path == "/api/sd15/generate":
            job_id = uuid.uuid4().hex[:8]
            mode = payload.get("mode", "full")
            if mode not in {"manual", "full"}:
                self.send_json({"error": f"Unsupported SD1.5 mode: {mode}"}, status=400)
                return

            try:
                with SD_JOBS_LOCK:
                    if any_sd_running():
                        self.send_json({"error": "An SD1.5 generation is already running."}, status=409)
                        return

                config_path = build_sd15_job_config(job_id, payload, mode)
                cmd_parts = build_sd15_command(config_path, mode)

                with SD_JOBS_LOCK:
                    SD_JOBS[job_id] = {
                        "id": job_id,
                        "status": "queued",
                        "mode": mode,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "config_path": str(config_path),
                        "output_dir": str(SD15_OUTPUT_DIR),
                        "upscaled_dir": str(SD15_UPSCALED_DIR),
                    }

                threading.Thread(target=run_sd_job, args=(job_id, cmd_parts, mode), daemon=True).start()
                self.send_json({"job_id": job_id, "message": f"SD1.5 {mode} pipeline started."})
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
            return
        if parsed.path == "/api/images/action":
            action = payload.get("action")
            filename = payload.get("filename")
            if action in ["like", "delete"] and filename:
                log_entry = {"timestamp": datetime.now().isoformat(), "filename": filename, "action": action}
                with open(IMAGES_LOG_PATH, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                if action == "delete":
                    path = (IMAGES_DIR / filename).resolve()
                    if path.parent == IMAGES_DIR.resolve() and path.exists():
                        os.remove(path)
                self.send_json({"status": "ok"})
            else:
                self.send_json({"error": "Invalid action or filename"}, status=400)
            return
        if parsed.path not in {"/api/generate", "/api/generate-queue"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        with JOBS_LOCK:
            if any_running():
                self.send_json({"error": "A generation is already running."}, status=409)
                return

        job_id = uuid.uuid4().hex[:12]
        try:
            if parsed.path == "/api/generate-queue":
                config_path, track_ids = build_queue_config(job_id, payload)
                track_id = f"Medium queue ({len(track_ids)} tracks)"
            else:
                config_path, track_id = build_config(job_id, payload)
                track_ids = [track_id]
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=400)
            return

        job = {
            "id": job_id,
            "track_id": track_id,
            "status": "queued",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "queue_count": len(track_ids),
        }
        with JOBS_LOCK:
            JOBS[job_id] = job

        thread = threading.Thread(target=run_job, args=(job_id, config_path), daemon=True)
        thread.start()
        self.send_json(
            {
                "job_id": job_id,
                "track_id": track_id,
                "track_ids": track_ids,
                "queue_count": len(track_ids),
            }
        )

# --- Main ---

def main() -> int:
    parser = argparse.ArgumentParser(description="LAN web UI for Stable Audio 3 continuation generation")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    WEB_RUNS.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    load_existing_jobs()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}", flush=True)
    print(f"LAN URL: http://{lan_ip()}:{args.port}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
