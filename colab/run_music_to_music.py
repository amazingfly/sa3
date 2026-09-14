#!/usr/bin/env python3
"""Run a Stable Audio 3 Medium music-to-music second pass on Colab."""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from run_generate import (
    install_sequential_chunked_decoder,
    install_streaming_medium_loader,
    patch_model_config_resolver,
    sha256_file,
    start_memory_monitor,
    validate_audio,
)


CONFIG_PATH = Path(
    os.environ.get("SA3_M2M_CONFIG", "/content/sa3/colab/music_to_music_config.json")
)
OUTPUT_ROOT = Path(os.environ.get("SA3_M2M_OUTPUT_ROOT", "/content/outputs/2nndPass"))
DEFAULT_INPUT_DIR = Path("/content/sa3/musicLibrary/flac")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def safe_name(value: str, *, field: str) -> str:
    name = Path(value).name
    if name != value or not name:
        raise ValueError(f"{field} must be a plain filename")
    return name


def read_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Music-to-music config not found: {CONFIG_PATH}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Music-to-music config must be a JSON object")

    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    if model.get("name") != "medium":
        raise ValueError("music-to-music second pass currently supports model.name='medium'")
    if not model.get("repo"):
        raise ValueError("model.repo is required")

    generation = config.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object")
    if not str(generation.get("prompt") or "").strip():
        raise ValueError("generation.prompt is required")
    init_noise_level = float(generation.get("init_noise_level", 0.27))
    if not 0 <= init_noise_level <= 1:
        raise ValueError("generation.init_noise_level must be between 0 and 1")
    if int(generation.get("steps", 12)) < 1:
        raise ValueError("generation.steps must be positive")

    paths = config.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("paths must be an object")
    input_dir = Path(paths.get("input_dir") or DEFAULT_INPUT_DIR)
    flac_dir = Path(paths.get("flac_dir") or OUTPUT_ROOT / "flac")
    ogg_dir = Path(paths.get("ogg_dir") or OUTPUT_ROOT / "ogg")
    report_dir = Path(paths.get("report_dir") or OUTPUT_ROOT / "reports")
    paths["input_dir"] = str(input_dir)
    paths["flac_dir"] = str(flac_dir)
    paths["ogg_dir"] = str(ogg_dir)
    paths["report_dir"] = str(report_dir)

    if "tracks" in config:
        tracks = config["tracks"]
        if not isinstance(tracks, list) or not tracks:
            raise ValueError("tracks must be a non-empty list when provided")
        seen_outputs: set[str] = set()
        for index, track in enumerate(tracks, start=1):
            if not isinstance(track, dict):
                raise ValueError(f"tracks[{index - 1}] must be an object")
            source_name = str(track.get("source_name") or "")
            if not source_name:
                raise ValueError(f"tracks[{index - 1}].source_name is required")
            safe_name(source_name, field=f"tracks[{index - 1}].source_name")
            if not source_name.lower().endswith(".flac"):
                raise ValueError(f"tracks[{index - 1}].source_name must end in .flac")
            output_name = str(track.get("output_name") or source_name)
            safe_name(output_name, field=f"tracks[{index - 1}].output_name")
            if not output_name.lower().endswith(".flac"):
                raise ValueError(f"tracks[{index - 1}].output_name must end in .flac")
            if output_name in seen_outputs:
                raise ValueError(f"Duplicate output_name: {output_name}")
            seen_outputs.add(output_name)

    return config


def discover_tracks(config: dict[str, Any]) -> list[dict[str, Any]]:
    if "tracks" in config:
        return [
            {
                **track,
                "source_name": safe_name(str(track["source_name"]), field="source_name"),
                "output_name": safe_name(
                    str(track.get("output_name") or track["source_name"]),
                    field="output_name",
                ),
            }
            for track in config["tracks"]
        ]

    input_dir = Path(config["paths"]["input_dir"])
    pattern = str(config.get("source_pattern") or "*.flac")
    tracks = []
    for source_path in sorted(input_dir.glob(pattern)):
        if source_path.is_file() and source_path.suffix.lower() == ".flac":
            tracks.append(
                {
                    "source_name": source_path.name,
                    "output_name": source_path.name,
                }
            )
    if not tracks:
        raise FileNotFoundError(f"No source FLAC files found in {input_dir}")
    return tracks


def audio_metadata(path: Path, expected_duration: float | None = None) -> dict[str, Any]:
    import soundfile as sf

    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise RuntimeError(f"Invalid audio metadata for {path}: {info}")

    duration = info.frames / info.samplerate
    if expected_duration is not None and abs(duration - expected_duration) > 0.15:
        raise RuntimeError(
            f"Unexpected duration for {path}: {duration:.3f}s; "
            f"expected {expected_duration:.3f}s"
        )

    peak = 0.0
    rms_sum = 0.0
    sample_count = 0
    with sf.SoundFile(str(path)) as audio_file:
        while True:
            block = audio_file.read(65536, dtype="float32", always_2d=True)
            if not len(block):
                break
            if not np.isfinite(block).all():
                raise RuntimeError(f"{path} contains NaN or infinite samples")
            peak = max(peak, float(np.max(np.abs(block))))
            rms_sum += float(np.square(block, dtype=np.float64).sum())
            sample_count += block.size

    rms = math.sqrt(rms_sum / sample_count) if sample_count else 0.0
    if peak <= 1e-5 or rms <= 1e-6:
        raise RuntimeError(f"{path} is effectively silent: peak={peak}, rms={rms}")

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_seconds": round(duration, 6),
        "peak": round(peak, 8),
        "rms": round(rms, 8),
        "sha256": sha256_file(path),
    }


def convert_flac_to_ogg(source: Path, destination: Path, quality: int) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.ogg")
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-c:a",
                "libvorbis",
                "-q:a",
                str(quality),
                str(temporary),
            ],
            check=True,
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg did not create a valid OGG: {temporary}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return audio_metadata(destination)


def source_duration_seconds(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


def planned_duration(
    track: dict[str, Any],
    generation: dict[str, Any],
    source_path: Path,
    max_duration: float,
) -> float:
    raw_duration = track.get("duration_seconds", generation.get("duration_seconds", "source"))
    if raw_duration in (None, "", "source"):
        duration = source_duration_seconds(source_path)
    else:
        duration = float(raw_duration)
    if duration <= 0:
        raise ValueError(f"Duration must be positive for {source_path.name}")
    if duration > max_duration:
        if generation.get("duration_policy", "clamp") == "clamp":
            return max_duration
        raise ValueError(
            f"{source_path.name} duration {duration:g}s exceeds max {max_duration:g}s"
        )
    return duration


def load_init_audio(source_path: Path, duration: float, target_sample_rate: int):
    import soundfile as sf
    import torch
    import torchaudio.functional as F

    source_info = sf.info(str(source_path))
    source_frames = min(source_info.frames, int(round(duration * source_info.samplerate)))
    data, source_rate = sf.read(
        str(source_path),
        frames=source_frames,
        dtype="float32",
        always_2d=True,
    )
    if data.size == 0:
        raise RuntimeError(f"Source audio is empty: {source_path}")

    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]

    audio = torch.from_numpy(np.ascontiguousarray(data.T))
    if int(source_rate) != int(target_sample_rate):
        audio = F.resample(audio, int(source_rate), int(target_sample_rate))

    target_samples = int(round(duration * target_sample_rate))
    if audio.shape[-1] < target_samples:
        pad = target_samples - audio.shape[-1]
        audio = torch.nn.functional.pad(audio, (0, pad))
    else:
        audio = audio[:, :target_samples]
    return audio.contiguous()


def install_attention_fallback_if_needed(model_name: str, model_config: dict[str, Any]) -> str:
    import torch

    props = torch.cuda.get_device_properties(0)
    allow_turing_fallback = bool(model_config.get("allow_turing_sdpa_fallback", False))
    if model_name == "medium" and props.major < 8 and not allow_turing_fallback:
        raise RuntimeError("Medium on Turing requires allow_turing_sdpa_fallback=true")
    if model_name == "medium" and props.major < 8 and allow_turing_fallback:
        import stable_audio_3.models.transformer as transformer_module

        transformer_module.flex_attention_compiled = None
        return "chunked-halo-sdpa"
    if model_name == "medium" and props.major >= 8:
        import flash_attn  # noqa: F401

        return "flash-attn-2"
    return "default"


def main() -> int:
    import soundfile as sf
    import torch
    from stable_audio_3 import StableAudioModel

    config = read_config()
    tracks = discover_tracks(config)
    model_config = config["model"]
    generation = config["generation"]
    paths = config["paths"]

    input_dir = Path(paths["input_dir"])
    flac_dir = Path(paths["flac_dir"])
    ogg_dir = Path(paths["ogg_dir"])
    report_dir = Path(paths["report_dir"])
    work_dir = OUTPUT_ROOT / "work"
    for directory in (flac_dir, ogg_dir, report_dir, work_dir, OUTPUT_ROOT):
        directory.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this in a Colab GPU runtime")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()

    model_name = str(model_config["name"])
    model_repo = str(model_config["repo"])
    conditioning_repo = str(model_config.get("conditioning_repo") or model_repo)
    attention_backend = install_attention_fallback_if_needed(model_name, model_config)
    patch_model_config_resolver(
        model_name,
        model_repo,
        conditioning_repo,
        work_dir,
    )
    if model_name == "medium" and model_config.get("low_memory_load", True):
        install_streaming_medium_loader()

    finish_memory_monitor = start_memory_monitor()
    started = time.monotonic()
    error_messages: list[str] = []
    results: list[dict[str, Any]] = []

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_name} from {model_repo}")
    print(f"Music-to-music source tracks: {len(tracks)}")
    print("Loading model once for the complete second-pass queue", flush=True)
    model_load_started = time.monotonic()
    model = StableAudioModel.from_pretrained(
        model_name,
        device="cuda",
        model_half=bool(model_config.get("model_half", True)),
    )
    if model_config.get("sequential_chunked_decode", False):
        install_sequential_chunked_decoder(
            model,
            chunk_size=int(model_config.get("decode_chunk_size", 64)),
            overlap=int(model_config.get("decode_overlap", 32)),
        )
    model_load_seconds = time.monotonic() - model_load_started
    sample_rate = int(model.model.sample_rate)
    model_sample_size = int(model.model_config.get("sample_size", 16_777_216))
    max_duration = float(model_config.get("max_duration_seconds", 380))

    seed_start = int(generation.get("seed_start", generation.get("seed", 620000)))
    steps = int(generation.get("steps", 12))
    cfg_scale = float(generation.get("cfg_scale", 1.0))
    apg_scale = float(generation.get("apg_scale", 1.0))
    init_noise_level = float(generation.get("init_noise_level", 0.27))
    duration_padding = float(generation.get("duration_padding_seconds", 6.0))
    ogg_quality = int(config.get("ogg_quality", generation.get("ogg_quality", 6)))
    skip_existing = bool(config.get("skip_existing", True))

    for index, track in enumerate(tracks, start=1):
        item_started = time.monotonic()
        source_name = str(track["source_name"])
        output_name = str(track.get("output_name") or source_name)
        source_path = input_dir / source_name
        flac_path = flac_dir / output_name
        ogg_path = ogg_dir / Path(output_name).with_suffix(".ogg").name
        report_path = report_dir / Path(output_name).with_suffix(".json").name
        seed = int(track.get("seed", seed_start + index - 1))
        torch.cuda.reset_peak_memory_stats()

        try:
            if not source_path.is_file():
                raise FileNotFoundError(f"Source FLAC not found: {source_path}")
            duration = planned_duration(track, generation, source_path, max_duration)
            target_samples = int(round(duration * sample_rate))
            if skip_existing and flac_path.exists() and ogg_path.exists() and report_path.exists():
                print(f"Skipping existing second-pass output: {output_name}", flush=True)
                result = json.loads(report_path.read_text(encoding="utf-8"))
                results.append(result)
                continue
            if flac_path.exists() or ogg_path.exists():
                raise FileExistsError(f"Refusing to overwrite output: {output_name}")

            requested_sample_size = int((duration + duration_padding + 12.0) * sample_rate)
            sample_size = min(model_sample_size, requested_sample_size)
            init_audio = load_init_audio(source_path, duration, sample_rate)
            source_meta = audio_metadata(source_path)
            print(
                f"Second-pass item {index}/{len(tracks)}: {source_name} -> {output_name}, "
                f"duration={duration:g}s, steps={steps}, cfg={cfg_scale}, "
                f"init_noise={init_noise_level}, seed={seed}",
                flush=True,
            )

            audio = model.generate(
                prompt=str(track.get("prompt") or generation["prompt"]),
                negative_prompt=track.get("negative_prompt", generation.get("negative_prompt")),
                duration=duration,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                batch_size=1,
                chunked_decode=bool(model_config.get("chunked_decode", True)),
                apg_scale=apg_scale,
                duration_padding_sec=duration_padding,
                sample_size=sample_size,
                sampler_type=str(generation.get("sampler_type", "pingpong")),
                init_audio=(sample_rate, init_audio),
                init_noise_level=init_noise_level,
            )
            print(
                f"Second-pass item {index}/{len(tracks)} decode returned; moving audio to CPU",
                flush=True,
            )
            audio = audio.detach().to(dtype=torch.float32, device="cpu")
            if audio.dim() == 3:
                audio = audio[0]
            if audio.dim() != 2:
                raise RuntimeError(f"Unexpected audio tensor shape: {tuple(audio.shape)}")
            audio = audio[:, :target_samples].clamp(-1, 1).contiguous()
            sf.write(
                str(flac_path),
                audio.transpose(0, 1).numpy(),
                sample_rate,
                format="FLAC",
                subtype=str(generation.get("flac_subtype", "PCM_24")),
            )
            flac_meta = validate_audio(flac_path, duration)
            ogg_meta = convert_flac_to_ogg(flac_path, ogg_path, ogg_quality)
            result = {
                "ok": True,
                "index": index,
                "id": track.get("id") or Path(source_name).stem,
                "elapsed_seconds": round(time.monotonic() - item_started, 3),
                "source": source_meta,
                "prompt": str(track.get("prompt") or generation["prompt"]),
                "negative_prompt": track.get("negative_prompt", generation.get("negative_prompt")),
                "settings": {
                    "duration_seconds": duration,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "apg_scale": apg_scale,
                    "seed": seed,
                    "sampler_type": str(generation.get("sampler_type", "pingpong")),
                    "init_noise_level": init_noise_level,
                    "chunked_decode": bool(model_config.get("chunked_decode", True)),
                    "sequential_chunked_decode": bool(
                        model_config.get("sequential_chunked_decode", False)
                    ),
                    "sample_size": sample_size,
                    "duration_padding_seconds": duration_padding,
                    "flac_subtype": str(generation.get("flac_subtype", "PCM_24")),
                    "ogg_quality": ogg_quality,
                    "peak_cuda_memory_gib": round(
                        torch.cuda.max_memory_allocated() / 1024**3, 3
                    ),
                },
                "flac": flac_meta,
                "ogg": ogg_meta,
            }
            print(f"Second-pass FLAC: {flac_path}", flush=True)
            print(f"Second-pass OGG: {ogg_path}", flush=True)
        except Exception:
            error = traceback.format_exc()
            error_messages.append(f"{source_name}:\n{error}")
            result = {
                "ok": False,
                "index": index,
                "id": track.get("id") or Path(source_name).stem,
                "elapsed_seconds": round(time.monotonic() - item_started, 3),
                "source_name": source_name,
                "output_name": output_name,
                "error": error,
            }
            print(error, file=sys.stderr, flush=True)
        finally:
            try:
                del audio
            except UnboundLocalError:
                pass
            try:
                del init_audio
            except UnboundLocalError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        write_json_atomic(report_path, result)
        results.append(result)
        print(f"Second-pass item report: {report_path}", flush=True)

    elapsed = time.monotonic() - started
    memory = finish_memory_monitor()
    manifest = {
        "ok": all(result.get("ok") for result in results),
        "created_at": utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "model_load_seconds": round(model_load_seconds, 3),
        "model_load_count": 1,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config_path": str(CONFIG_PATH),
        "model_name": model_name,
        "model_repo": model_repo,
        "conditioning_repo": conditioning_repo,
        "attention_backend": attention_backend,
        "input_dir": str(input_dir),
        "flac_dir": str(flac_dir),
        "ogg_dir": str(ogg_dir),
        "report_dir": str(report_dir),
        "count": len(results),
        "succeeded": sum(bool(result.get("ok")) for result in results),
        "failed": sum(not result.get("ok") for result in results),
        "memory": memory,
        "results": results,
    }
    write_json_atomic(OUTPUT_ROOT / "music_to_music.json", manifest)
    if error_messages:
        (OUTPUT_ROOT / "music_to_music_error.log").write_text(
            "\n\n".join(error_messages),
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        error = traceback.format_exc()
        (OUTPUT_ROOT / "music_to_music_error.log").write_text(error, encoding="utf-8")
        print(error, file=sys.stderr)
        raise
