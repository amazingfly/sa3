#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def gb(value: float) -> float:
    return value / (1024**3)


def slugify(value: str) -> str:
    import re

    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "track"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if len(config.get("track", {}).get("acts", [])) != 3:
        raise ValueError("Continuation config must define exactly three acts.")
    return config


def monitor_call(fn: Callable[[], Any], interval_seconds: float) -> tuple[Any, dict[str, Any]]:
    import psutil  # type: ignore

    process = psutil.Process(os.getpid())
    stop = threading.Event()
    peak_rss = 0
    min_available = float("inf")
    samples = 0
    start = time.monotonic()

    def collect() -> None:
        nonlocal peak_rss, min_available, samples
        while not stop.is_set():
            rss = process.memory_info().rss
            peak_rss = max(peak_rss, rss)
            min_available = min(min_available, psutil.virtual_memory().available)
            samples += 1
            stop.wait(interval_seconds)

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join()
    peak_rss = max(peak_rss, process.memory_info().rss)
    min_available = min(min_available, psutil.virtual_memory().available)
    return result, {
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "peak_process_rss_gb": round(gb(float(peak_rss)), 3),
        "min_system_available_gb": round(gb(float(min_available)), 3),
        "samples": samples,
    }


def save_flac_and_ogg(audio, sample_rate: int, flac_path: Path, ogg_path: Path | None = None) -> None:
    import torchaudio  # type: ignore

    flac_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(flac_path), audio.cpu(), sample_rate)
    if ogg_path:
        ogg_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-i",
                str(flac_path),
                "-c:a",
                "libvorbis",
                "-q:a",
                "6",
                str(ogg_path),
            ],
            check=True,
        )


def run(config: dict[str, Any]) -> int:
    import torch  # type: ignore
    from stable_audio_3 import StableAudioModel  # type: ignore
    from scripts.run_sa3_workflow import patch_model_config_resolver

    cpu_cfg = config.get("cpu", {})
    if cpu_cfg.get("torch_num_threads"):
        torch.set_num_threads(int(cpu_cfg["torch_num_threads"]))
    if cpu_cfg.get("torch_num_interop_threads"):
        torch.set_num_interop_threads(int(cpu_cfg["torch_num_interop_threads"]))

    model_cfg = config["model"]
    run_dir = ROOT / config.get("output_dir", "outputs") / f"{timestamp()}-continuation"
    audio_dir = run_dir / "audio"
    jobs_dir = run_dir / "jobs"
    audio_dir.mkdir(parents=True, exist_ok=False)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    library_dir = ROOT / config.get("music_library_dir", "musicLibrary")
    run_name = run_dir.name
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "config": config,
        "acts": [],
    }

    conditioning_repo_override = model_cfg.get("conditioning_repo_id_override")
    if conditioning_repo_override:
        patch_model_config_resolver(
            model_name=model_cfg["name"],
            repo_id=model_cfg["hf_repo"],
            conditioning_repo_id=conditioning_repo_override,
            work_dir=jobs_dir,
        )

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Loading model {model_cfg['name']} from {model_cfg['hf_repo']}", flush=True)
    model = StableAudioModel.from_pretrained(
        model_cfg["name"],
        device=model_cfg.get("device", "cpu"),
        model_half=bool(model_cfg.get("model_half", False)),
    )

    generation = config["generation"]
    duration = float(generation["duration_seconds"])
    overlap_seconds = float(generation.get("overlap_anchor_seconds", 8))
    seed_start = int(generation.get("seed_start", 350000))
    sample_rate = int(model.model.sample_rate)
    target_samples = int(duration * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)

    track = config["track"]
    segments = []

    for index, act in enumerate(track["acts"]):
        act_id = slugify(act["id"])
        flac_path = audio_dir / f"{slugify(track['id'])}-{index + 1}-{act_id}-{int(duration)}s.flac"
        seed = seed_start + index

        kwargs: dict[str, Any] = {
            "prompt": act["prompt"],
            "negative_prompt": generation.get("negative_prompt"),
            "duration": duration,
            "steps": int(generation.get("steps", 8)),
            "cfg_scale": float(generation.get("cfg_scale", 1.0)),
            "seed": seed,
            "batch_size": 1,
            "chunked_decode": bool(model_cfg.get("chunked_decode", True)),
            "apg_scale": float(generation.get("apg_scale", 1.0)),
            "duration_padding_sec": float(generation.get("duration_padding_sec", 6.0)),
        }
        if generation.get("sampler_type"):
            kwargs["sampler_type"] = generation["sampler_type"]

        if index > 0:
            source = torch.zeros((segments[-1].shape[0], target_samples), dtype=segments[-1].dtype)
            source[:, :overlap_samples] = segments[-1][:, -overlap_samples:]
            kwargs["inpaint_audio"] = (sample_rate, source)
            kwargs["inpaint_mask_start_seconds"] = overlap_seconds
            kwargs["inpaint_mask_end_seconds"] = duration

        print(
            f"Generating act {index + 1}/3: {act_id}, duration={duration:g}s, "
            f"steps={kwargs['steps']}, cfg={kwargs['cfg_scale']}, seed={seed}",
            flush=True,
        )

        def generate_one():
            audio = model.generate(**kwargs).detach().to(torch.float32).cpu()
            return audio[0] if audio.dim() == 3 else audio

        audio, memory = monitor_call(
            generate_one, float(config.get("memory", {}).get("sample_interval_seconds", 0.5))
        )
        audio = audio[:, :target_samples].clamp(-1, 1)
        segments.append(audio)
        save_flac_and_ogg(audio, sample_rate, flac_path)

        result = {
            "id": act["id"],
            "seed": seed,
            "prompt": act["prompt"],
            "output_flac": str(flac_path),
            "memory": memory,
        }
        manifest["acts"].append(result)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"Finished act {index + 1}: elapsed={memory['elapsed_seconds']}s, "
            f"peak_rss={memory['peak_process_rss_gb']} GiB, "
            f"min_available={memory['min_system_available_gb']} GiB",
            flush=True,
        )

    combined = torch.cat(segments, dim=-1).clamp(-1, 1)
    combined_flac = audio_dir / f"{slugify(track['id'])}-combined-360s.flac"
    combined_ogg = library_dir / "ogg" / f"{run_name}__{combined_flac.stem}.ogg"
    save_flac_and_ogg(combined, sample_rate, combined_flac, combined_ogg)
    manifest["combined"] = {
        "output_flac": str(combined_flac),
        "library_ogg": str(combined_ogg),
        "duration_seconds": combined.shape[-1] / sample_rate,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved combined 360s track: {combined_flac}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stable Audio 3 continuation suite runner")
    parser.add_argument("--config", default="configs/cybermetal_electronica_360_continuation.json")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    return run(load_config(config_path))


if __name__ == "__main__":
    raise SystemExit(main())
