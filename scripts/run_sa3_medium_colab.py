#!/usr/bin/env python3
"""Adapt web SA3 Medium jobs to the Colab CLI generation workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
MEDIUM_REPO = "cocktailpeanut/stable-audio-3-medium"


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "medium-track"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    return config


def first_track(config: dict[str, Any]) -> dict[str, Any]:
    tracks = config.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        return {}
    if len(tracks) > 1:
        raise ValueError("SA3 Medium Colab web jobs currently support one track")
    return tracks[0] if isinstance(tracks[0], dict) else {}


def require_medium(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    if not isinstance(model, dict) or model.get("name") != "medium":
        raise ValueError("The Colab web adapter only supports model.name='medium'")
    return model


def normalize_config(config: dict[str, Any], job_id: str) -> tuple[dict[str, Any], str]:
    """Accept native Colab or legacy web config keys and produce one Colab config."""
    model = require_medium(config)
    generation = config.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("Config must contain a generation object")
    track = first_track(config)

    prompt = str(
        generation.get("prompt")
        or config.get("prompt")
        or track.get("prompt")
        or ""
    ).strip()
    if not prompt:
        raise ValueError(
            "Medium prompt is required in generation.prompt, prompt, or tracks[0].prompt"
        )

    track_id = slugify(
        str(config.get("track_id") or track.get("id") or f"medium-{job_id}")
    )
    duration = float(generation.get("duration_seconds", 380))
    if not 10 <= duration <= 380:
        raise ValueError("SA3 Medium duration_seconds must be between 10 and 380")
    steps = int(generation.get("steps", 8))
    if not 1 <= steps <= 80:
        raise ValueError("SA3 Medium steps must be between 1 and 80")

    seed_value = generation.get("seed", generation.get("seed_start"))
    seed = int(seed_value) if seed_value not in (None, "") else 360606
    padding = float(
        generation.get(
            "duration_padding_seconds",
            generation.get("duration_padding_sec", 6.0),
        )
    )
    repo = str(model.get("repo") or model.get("hf_repo") or MEDIUM_REPO)
    conditioning_repo = str(
        model.get("conditioning_repo")
        or model.get("conditioning_repo_id_override")
        or repo
    )

    requested_name = Path(str(generation.get("output_name") or "")).name
    if requested_name and Path(requested_name).suffix.lower() != ".flac":
        raise ValueError("SA3 Medium output_name must end in .flac")
    job_slug = slugify(job_id)
    if requested_name:
        output_stem = slugify(Path(requested_name).stem)
        if not output_stem.endswith(f"-{job_slug}"):
            output_stem = f"{output_stem}-{job_slug}"
        output_name = f"{output_stem}.flac"
    else:
        output_name = f"{track_id}-medium-{int(duration)}s-{job_slug}.flac"

    normalized = {
        "model": {
            "name": "medium",
            "repo": repo,
            "conditioning_repo": conditioning_repo,
            "model_half": True,
            "low_memory_load": True,
            "chunked_decode": bool(model.get("chunked_decode", True)),
            "decode_chunk_size": int(model.get("decode_chunk_size", 64)),
            "decode_overlap": int(model.get("decode_overlap", 32)),
            "allow_turing_sdpa_fallback": True,
            "max_duration_seconds": 380,
        },
        "generation": {
            "duration_seconds": duration,
            "steps": steps,
            "cfg_scale": float(generation.get("cfg_scale", 1.0)),
            "apg_scale": float(generation.get("apg_scale", 1.0)),
            "duration_padding_seconds": padding,
            "sampler_type": str(generation.get("sampler_type", "pingpong")),
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": generation.get("negative_prompt"),
            "output_name": output_name,
            "flac_subtype": str(generation.get("flac_subtype", "PCM_24")),
        },
    }
    return normalized, track_id


def normalize_source(
    source_config: dict[str, Any], job_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    items = source_config.get("items")
    if items is None:
        normalized, track_id = normalize_config(source_config, job_id)
        return normalized, [source_config], [track_id]
    if source_config.get("queue_version") != 1:
        raise ValueError("Medium queue config must set queue_version to 1")
    if not isinstance(items, list) or not items:
        raise ValueError("Medium queue items must be a non-empty list")
    if len(items) > 20:
        raise ValueError("Medium queue supports at most 20 items")

    normalized_items = []
    source_items = []
    track_ids = []
    shared_model = None
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Queue item {index} must be an object")
        normalized, track_id = normalize_config(
            item, f"{job_id}-{index:03d}"
        )
        model = normalized["model"]
        if shared_model is None:
            shared_model = model
        elif model != shared_model:
            raise ValueError(
                "All Medium queue items must use the same model configuration"
            )
        generation = dict(normalized["generation"])
        generation["id"] = track_id
        normalized_items.append(generation)
        source_items.append(item)
        track_ids.append(track_id)

    return (
        {
            "queue_version": 1,
            "model": shared_model,
            "generations": normalized_items,
        },
        source_items,
        track_ids,
    )


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing library audio: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def convert_flac_to_ogg(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to create the music library OGG")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing OGG audio: {destination}")
    temporary = destination.with_name(f".{destination.stem}.tmp.ogg")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(source),
                "-c:a",
                "libvorbis",
                "-q:a",
                "6",
                str(temporary),
            ],
            check=True,
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg did not create a valid OGG: {temporary}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_checkpoint(generation: dict[str, Any], download_dir: Path) -> bool:
    output_name = generation["output_name"]
    output_path = download_dir / output_name
    report_path = download_dir / Path(output_name).with_suffix(".json")
    if not output_path.is_file() or not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        audio = report["audio"]
        info = sf.info(str(output_path))
        return (
            bool(report.get("ok"))
            and output_path.stat().st_size == int(audio["bytes"])
            and sha256_file(output_path) == audio["sha256"]
            and abs(info.duration - float(audio["duration_seconds"])) < 0.01
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_web_manifest(
    source_configs: list[dict[str, Any]],
    normalized: dict[str, Any],
    track_ids: list[str],
    run_dir: Path,
    outputs: list[tuple[Path, Path, Path]],
    generation_report: dict[str, Any],
) -> Path:
    generations = normalized.get("generations") or [normalized["generation"]]
    reports = generation_report.get("results") or [generation_report]
    tracks = []
    for source, generation, track_id, paths, report in zip(
        source_configs,
        generations,
        track_ids,
        outputs,
        reports,
        strict=True,
    ):
        output_flac, library_flac, library_ogg = paths
        combined = {
            "output_flac": str(output_flac),
            "library_flac": str(library_flac),
            "library_ogg": str(library_ogg),
            "duration_seconds": report["audio"]["duration_seconds"],
        }
        tracks.append(
            {
                "id": track_id,
                "config": source,
                "acts": [
                    {
                        "id": "direct",
                        "seed": generation["seed"],
                        "prompt": generation["prompt"],
                        "output_flac": str(output_flac),
                        "library_flac": str(library_flac),
                        "library_ogg": str(library_ogg),
                        "memory": report.get("memory", {}),
                    }
                ],
                "combined": combined,
            }
        )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "mode": "direct",
        "backend": "colab-cli",
        "queue": len(tracks) > 1,
        "config": {
            "queue_version": 1,
            "items": source_configs,
        },
        "colab_config": normalized,
        "colab_generation": generation_report,
        "tracks": tracks,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def downloaded_generation_report(
    normalized: dict[str, Any],
    download_dir: Path,
    attempts: int,
) -> dict[str, Any]:
    generations = normalized.get("generations") or [normalized["generation"]]
    results = []
    for index, generation in enumerate(generations, start=1):
        output_name = generation["output_name"]
        output_path = download_dir / output_name
        report_path = download_dir / Path(output_name).with_suffix(".json")
        if report_path.is_file():
            result = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            info = sf.info(str(output_path))
            result = {
                "ok": True,
                "index": index,
                "id": generation.get("id") or f"item-{index}",
                "prompt": generation["prompt"],
                "negative_prompt": generation.get("negative_prompt"),
                "settings": {
                    key: generation.get(key)
                    for key in (
                        "duration_seconds",
                        "steps",
                        "cfg_scale",
                        "apg_scale",
                        "seed",
                        "sampler_type",
                        "flac_subtype",
                    )
                },
                "audio": {
                    "path": str(output_path),
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "frames": info.frames,
                    "duration_seconds": round(info.duration, 6),
                    "format": info.format,
                    "subtype": info.subtype,
                },
            }
        result["index"] = index
        result["audio"]["path"] = str(output_path)
        results.append(result)
    return {
        "ok": all(result.get("ok") for result in results),
        "queue": len(results) > 1,
        "count": len(results),
        "succeeded": sum(bool(result.get("ok")) for result in results),
        "failed": sum(not result.get("ok") for result in results),
        "resume_attempts": attempts,
        "model_name": normalized["model"]["name"],
        "model_repo": normalized["model"]["repo"],
        "results": results,
    }


def run(
    config_path: Path,
    job_id: str,
    normalize_only: bool = False,
    item_index: int | None = None,
    resume_run_dir: Path | None = None,
) -> int:
    source_config = read_config(config_path)
    if item_index is not None:
        items = source_config.get("items")
        if not isinstance(items, list) or not 1 <= item_index <= len(items):
            raise ValueError(
                f"--item-index must select one of {len(items or [])} queue items"
            )
        source_config = items[item_index - 1]
    normalized, source_configs, track_ids = normalize_source(source_config, job_id)
    if normalize_only:
        print(json.dumps(normalized, indent=2))
        return 0

    first_source = source_configs[0]
    output_root = Path(str(first_source.get("output_dir") or ROOT / "outputs"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    library_root = Path(
        str(first_source.get("music_library_dir") or ROOT / "musicLibrary")
    )
    if not library_root.is_absolute():
        library_root = ROOT / library_root

    if resume_run_dir is None:
        run_dir = output_root / f"{timestamp()}-web-medium-colab-{slugify(job_id)}"
    else:
        run_dir = resume_run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory not found: {run_dir}")
    jobs_dir = run_dir / "jobs"
    audio_dir = run_dir / "audio"
    download_dir = run_dir / "colab"
    jobs_dir.mkdir(parents=True, exist_ok=resume_run_dir is not None)
    audio_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    label = "Resuming run directory" if resume_run_dir is not None else "Run directory"
    print(f"{label}: {run_dir}", flush=True)

    normalized_path = jobs_dir / "colab-generation-config.json"
    if resume_run_dir is not None and normalized_path.is_file():
        previous = json.loads(normalized_path.read_text(encoding="utf-8"))
        previous_generations = previous.get("generations") or [previous["generation"]]
        current_generations = normalized.get("generations") or [normalized["generation"]]
        previous_names = [item["output_name"] for item in previous_generations]
        current_names = [item["output_name"] for item in current_generations]
        if previous_names != current_names:
            raise ValueError(
                "Resume config output names do not match the existing run"
            )
    normalized_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    generations = normalized.get("generations") or [normalized["generation"]]
    max_attempts = int(os.environ.get("SA3_COLAB_MAX_ATTEMPTS", "16"))
    attempts = 0
    if resume_run_dir is not None:
        for path in jobs_dir.glob("colab-generation-config-attempt-*.json"):
            match = re.search(r"attempt-(\d+)\.json$", path.name)
            if match:
                attempts = max(attempts, int(match.group(1)))
        if attempts:
            print(
                f"Continuing after {attempts} previous Colab attempt(s)",
                flush=True,
            )
    while attempts < max_attempts:
        for generation in generations:
            output_name = generation["output_name"]
            output_path = download_dir / output_name
            report_path = download_dir / Path(output_name).with_suffix(".json")
            if output_path.exists() and not valid_checkpoint(
                generation,
                download_dir,
            ):
                print(
                    f"Removing invalid checkpoint: {output_name}",
                    flush=True,
                )
                output_path.unlink(missing_ok=True)
                report_path.unlink(missing_ok=True)
        pending = [
            generation
            for generation in generations
            if not valid_checkpoint(generation, download_dir)
        ]
        if not pending:
            break
        attempts += 1
        attempt_config = {
            "queue_version": 1,
            "model": normalized["model"],
            "generations": pending,
        }
        attempt_path = jobs_dir / (
            f"colab-generation-config-attempt-{attempts}.json"
        )
        attempt_path.write_text(
            json.dumps(attempt_config, indent=2),
            encoding="utf-8",
        )
        print(
            f"Colab attempt {attempts}/{max_attempts}: "
            f"{len(pending)} queue item(s) remaining",
            flush=True,
        )
        session_slug = re.sub(r"[^a-z0-9-]", "-", job_id.lower())[:24]
        env = os.environ.copy()
        env.update(
            {
                "COLAB_SESSION": f"sa3-web-{session_slug}-a{attempts}",
                "SA3_COLAB_CONFIG": str(attempt_path),
                "SA3_COLAB_LOCAL_OUTPUT_DIR": str(download_dir),
            }
        )
        completed = subprocess.run(
            [str(ROOT / "colab" / "run_colab.sh")],
            cwd=str(ROOT),
            env=env,
            check=False,
        )
        remaining = [
            generation["output_name"]
            for generation in generations
            if not valid_checkpoint(generation, download_dir)
        ]
        if not remaining:
            break
        print(
            f"Colab attempt {attempts} exited with code "
            f"{completed.returncode}; checkpointed "
            f"{len(generations) - len(remaining)}/{len(generations)} tracks",
            flush=True,
        )

    missing = [
        generation["output_name"]
        for generation in generations
        if not valid_checkpoint(generation, download_dir)
    ]
    if missing:
        raise RuntimeError(
            f"Colab did not finish after {attempts} attempts: "
            + ", ".join(missing)
        )
    generation_report = downloaded_generation_report(
        normalized,
        download_dir,
        attempts,
    )
    reports = generation_report["results"]

    outputs = []
    failures = []
    for generation, report in zip(generations, reports, strict=True):
        output_name = generation["output_name"]
        downloaded_flac = download_dir / output_name
        if not report.get("ok") or not downloaded_flac.is_file():
            failures.append(output_name)
            continue
        output_flac = audio_dir / output_name
        library_flac = library_root / "flac" / output_name
        library_ogg = library_root / "ogg" / Path(output_name).with_suffix(".ogg")
        if library_flac.exists() or library_ogg.exists():
            raise FileExistsError(
                "Refusing to overwrite existing library audio for "
                f"{Path(output_name).stem}"
            )
        shutil.move(str(downloaded_flac), output_flac)
        try:
            convert_flac_to_ogg(output_flac, library_ogg)
            hardlink_or_copy(output_flac, library_flac)
        except Exception:
            library_ogg.unlink(missing_ok=True)
            library_flac.unlink(missing_ok=True)
            raise
        outputs.append((output_flac, library_flac, library_ogg))
        print(f"Web FLAC: {library_flac}", flush=True)
        print(f"Web OGG: {library_ogg}", flush=True)

    if failures:
        raise RuntimeError(
            "Colab queue failed for: " + ", ".join(failures)
        )
    (download_dir / "sa3-colab-outputs.tar.gz").unlink(missing_ok=True)
    write_web_manifest(
        source_configs,
        normalized,
        track_ids,
        run_dir,
        outputs,
        generation_report,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an SA3 Medium web job on Colab")
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--normalize-only", action="store_true")
    parser.add_argument("--item-index", type=int)
    parser.add_argument("--resume-run-dir", type=Path)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    return run(
        config_path,
        args.job_id,
        normalize_only=args.normalize_only,
        item_index=args.item_index,
        resume_run_dir=args.resume_run_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
