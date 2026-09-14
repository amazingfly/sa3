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
    if not config.get("tracks"):
        raise ValueError("Config must define tracks.")
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
            peak_rss = max(peak_rss, process.memory_info().rss)
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


def prompt_for_act(base_prompt: str, act: dict[str, Any], standard: dict[str, str]) -> str:
    prefix = standard.get("base_prefix", "")
    suffix = standard.get(act["id"], "")
    parts = [prefix, base_prompt, suffix]
    return " ".join(part.strip() for part in parts if part and part.strip())


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
    run_dir = ROOT / config.get("output_dir", "outputs") / f"{timestamp()}-multi-continuation"
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
        "tracks": [],
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
    acts = config["acts"]
    standard_prompts = config.get("standard_prompts", {})
    duration = float(generation["duration_seconds"])
    seed = int(generation.get("seed_start", 360000))
    sample_rate = int(model.model.sample_rate)
    target_samples = int(duration * sample_rate)
    init_noise_level = float(generation.get("init_noise_level", 0.78))

    for track_index, track in enumerate(config["tracks"], start=1):
        track_id = slugify(track["id"])
        track_result: dict[str, Any] = {"id": track["id"], "acts": []}
        segments = []
        previous = None

        print(f"Starting track {track_index}/{len(config['tracks'])}: {track_id}", flush=True)

        for act_index, act in enumerate(acts, start=1):
            act_id = slugify(act["id"])
            prompt = prompt_for_act(track["prompt"], act, standard_prompts)
            flac_path = audio_dir / f"{track_id}-{act_index}-{act_id}-{int(duration)}s.flac"

            kwargs: dict[str, Any] = {
                "prompt": prompt,
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
            if previous is not None:
                kwargs["init_audio"] = (sample_rate, previous)
                kwargs["init_noise_level"] = init_noise_level

            print(
                f"Generating {track_id} act {act_index}/3: {act_id}, "
                f"duration={duration:g}s, steps={kwargs['steps']}, cfg={kwargs['cfg_scale']}, seed={seed}"
                + (f", init_noise={init_noise_level}" if previous is not None else ""),
                flush=True,
            )

            def generate_one():
                audio = model.generate(**kwargs).detach().to(torch.float32).cpu()
                return audio[0] if audio.dim() == 3 else audio

            audio, memory = monitor_call(
                generate_one,
                float(config.get("memory", {}).get("sample_interval_seconds", 0.5)),
            )
            audio = audio[:, :target_samples].clamp(-1, 1)
            save_flac_and_ogg(audio, sample_rate, flac_path)

            result = {
                "id": act["id"],
                "seed": seed,
                "prompt": prompt,
                "used_full_previous_segment_as_init_audio": previous is not None,
                "init_noise_level": init_noise_level if previous is not None else None,
                "output_flac": str(flac_path),
                "memory": memory,
            }
            track_result["acts"].append(result)
            print(
                f"Finished {track_id} {act_id}: elapsed={memory['elapsed_seconds']}s, "
                f"peak_rss={memory['peak_process_rss_gb']} GiB, "
                f"min_available={memory['min_system_available_gb']} GiB",
                flush=True,
            )

            segments.append(audio)
            previous = audio
            seed += 1
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        combined = torch.cat(segments, dim=-1).clamp(-1, 1)
        combined_flac = audio_dir / f"{track_id}-combined-360s.flac"
        combined_ogg = library_dir / "ogg" / f"{run_name}__{combined_flac.stem}.ogg"
        save_flac_and_ogg(combined, sample_rate, combined_flac, combined_ogg)
        track_result["combined"] = {
            "output_flac": str(combined_flac),
            "library_ogg": str(combined_ogg),
            "duration_seconds": combined.shape[-1] / sample_rate,
        }
        manifest["tracks"].append(track_result)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved combined 360s track: {combined_flac}", flush=True)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stable Audio 3 multi-track full-init continuation runner")
    parser.add_argument("--config", default="configs/cybermetal_standard_360_full_init_continuation.json")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    return run(load_config(config_path))


if __name__ == "__main__":
    raise SystemExit(main())
