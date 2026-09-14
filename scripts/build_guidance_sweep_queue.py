#!/usr/bin/env python3
"""Build a Medium Colab queue by regenerating selected tracks with CFG sweeps."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REPO = "cocktailpeanut/stable-audio-3-medium"


def slugify(value: str, *, max_len: int = 96) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-") or "track"
    return value[:max_len].strip("-") or "track"


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def report_candidates(search_roots: list[Path]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            payload = load_json(path)
            if not payload:
                continue
            candidates = []
            audio_path = payload.get("audio", {}).get("path")
            if audio_path:
                candidates.append(Path(str(audio_path)).stem)
            if payload.get("output_name"):
                candidates.append(Path(str(payload["output_name"])).stem)
            if path.name != "generation.json":
                candidates.append(path.stem)
            for key in candidates:
                reports.setdefault(key, payload)
    return reports


def resolve_report(stem: str, reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [stem]
    if "__" in stem:
        candidates.append(stem.split("__", 1)[1])
    for candidate in candidates:
        if candidate in reports:
            return reports[candidate]
    raise KeyError(f"No source report found for selected track: {stem}")


def selected_tracks(selected_dir: Path, selected_track_args: list[str]) -> list[Path]:
    if selected_track_args:
        files = [Path(value).expanduser() for value in selected_track_args]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise ValueError(f"Selected track path does not exist: {missing[0]}")
    else:
        files = sorted(
            path
            for suffix in ("*.ogg", "*.flac")
            for path in selected_dir.glob(suffix)
            if path.is_file()
        )
    if not files:
        raise ValueError(f"No selected .ogg or .flac tracks found")
    deduped: dict[str, Path] = {}
    for path in files:
        deduped.setdefault(path.stem, path)
    return list(deduped.values())


def cfg_values(start: float, step: float, loops: int, minimum: float) -> list[float]:
    values = []
    for index in range(loops):
        values.append(round(max(minimum, start - step * index), 3))
    return values


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    selected = selected_tracks(Path(args.selected_dir).expanduser(), args.selected_track)
    reports = report_candidates([Path(path).expanduser() for path in args.report_root])
    cfgs = cfg_values(args.cfg_start, args.cfg_step, args.loops, args.cfg_min)
    boost_pattern = args.boost_pattern.lower().strip()
    generations = []
    source_tracks = []
    source_records = []

    for selected_path in selected:
        report = resolve_report(selected_path.stem, reports)
        prompt = str(report.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"Source report has no prompt: {selected_path.name}")
        settings = report.get("settings", {})
        audio = report.get("audio", {})
        duration = float(
            args.duration_seconds
            or settings.get("duration_seconds")
            or audio.get("duration_seconds")
            or 380
        )
        negative_prompt = report.get("negative_prompt")
        track_slug = slugify(selected_path.stem)
        boosted = bool(boost_pattern and boost_pattern in selected_path.stem.lower())
        repeats = args.boost_count if boosted else 1
        source_records.append(
            {
                "selected_path": selected_path,
                "settings": settings,
                "duration": duration,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "track_slug": track_slug,
                "repeats": repeats,
            }
        )
        source_tracks.append(
            {
                "selected_file": str(selected_path),
                "source_stem": selected_path.stem,
                "boosted": boosted,
                "repeats_per_loop": repeats,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
            }
        )

    for loop_index, cfg in enumerate(cfgs, start=1):
        for record in source_records:
            settings = record["settings"]
            for repeat_index in range(1, record["repeats"] + 1):
                queue_index = len(generations)
                generation_id = (
                    f"l{loop_index:02d}-r{repeat_index:02d}-{record['track_slug']}"
                )
                generations.append(
                    {
                        "id": generation_id,
                        "duration_seconds": record["duration"],
                        "steps": args.steps or int(settings.get("steps", 14)),
                        "cfg_scale": cfg,
                        "apg_scale": args.apg_scale,
                        "duration_padding_seconds": args.duration_padding_seconds,
                        "sampler_type": str(settings.get("sampler_type", "pingpong")),
                        "seed": args.seed_start + queue_index,
                        "prompt": record["prompt"],
                        "negative_prompt": record["negative_prompt"],
                        "output_name": f"{args.output_prefix}-{generation_id}.flac",
                        "flac_subtype": str(settings.get("flac_subtype", "PCM_24")),
                    }
                )

    return {
        "queue_version": 1,
        "model": {
            "name": "medium",
            "repo": args.model_repo,
            "conditioning_repo": args.conditioning_repo or args.model_repo,
            "model_half": True,
            "low_memory_load": True,
            "chunked_decode": True,
            "sequential_chunked_decode": False,
            "allow_turing_sdpa_fallback": True,
            "max_duration_seconds": 380,
        },
        "metadata": {
            "selected_dir": args.selected_dir,
            "loops": args.loops,
            "cfg_values": cfgs,
            "boost_pattern": args.boost_pattern,
            "boost_count": args.boost_count,
            "source_track_count": len(selected),
            "queue_item_count": len(generations),
            "source_tracks": source_tracks,
        },
        "generations": generations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Medium Colab guidance sweep queue from selected tracks."
    )
    parser.add_argument(
        "--selected-dir",
        default="musicLibrary/ogg/yes/do",
        help="Directory containing selected .ogg/.flac tracks",
    )
    parser.add_argument(
        "--selected-track",
        action="append",
        default=[],
        help="Exact selected .ogg/.flac path; repeatable. Overrides --selected-dir.",
    )
    parser.add_argument(
        "--report-root",
        action="append",
        default=[str(ROOT / "outputs"), "/mnt/storage/sa3_colab_tweak_export_85"],
        help="Root to search recursively for per-track JSON reports; repeatable",
    )
    parser.add_argument("--output", default="configs/selected_guidance_sweep_medium_queue.json")
    parser.add_argument("--output-prefix", default="guidesweep-medium-380s")
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--cfg-start", type=float, default=1.0)
    parser.add_argument("--cfg-step", type=float, default=0.05)
    parser.add_argument("--cfg-min", type=float, default=0.55)
    parser.add_argument("--boost-pattern", default="")
    parser.add_argument("--boost-count", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=705000)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--apg-scale", type=float, default=1.0)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--duration-padding-seconds", type=float, default=6.0)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--conditioning-repo", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.loops < 1:
        raise ValueError("--loops must be positive")
    if args.boost_count < 1:
        raise ValueError("--boost-count must be positive")
    config = build_config(args)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    boosted = [
        item["source_stem"]
        for item in config["metadata"]["source_tracks"]
        if item["boosted"]
    ]
    print(
        f"Wrote {len(config['generations'])} queued generations from "
        f"{config['metadata']['source_track_count']} source tracks to {output}"
    )
    print(f"CFG values: {config['metadata']['cfg_values']}")
    print(f"Boosted tracks: {boosted or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
