#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "audio"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare(config: dict[str, Any]) -> Path:
    dataset = config["dataset"]
    source_dir = resolve_path(dataset["source_dir"])
    prepared_dir = resolve_path(dataset["prepared_dir"])
    link_mode = dataset.get("link_mode", "symlink")
    default_caption = dataset["default_caption"].strip()

    if link_mode not in {"symlink", "copy"}:
        raise ValueError("dataset.link_mode must be symlink or copy")
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    prepared_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    seen_names: set[str] = set()

    for index, item in enumerate(dataset.get("files", []), start=1):
        source = source_dir / item["source"]
        if not source.exists():
            raise FileNotFoundError(source)
        if source.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio extension: {source}")

        stem = f"{index:02d}-{slugify(source.stem)}"
        if stem in seen_names:
            raise ValueError(f"Duplicate prepared stem: {stem}")
        seen_names.add(stem)

        audio_out = prepared_dir / f"{stem}{source.suffix.lower()}"
        caption_out = prepared_dir / f"{stem}.txt"
        caption = item.get("caption", default_caption).strip()

        if audio_out.exists() or audio_out.is_symlink():
            audio_out.unlink()
        if link_mode == "symlink":
            os.symlink(os.path.relpath(source, audio_out.parent), audio_out)
        else:
            shutil.copy2(source, audio_out)

        caption_out.write_text(caption + "\n", encoding="utf-8")
        manifest.append(
            {
                "source": str(source.relative_to(ROOT)),
                "audio": str(audio_out.relative_to(ROOT)),
                "caption": str(caption_out.relative_to(ROOT)),
            }
        )

    if len(manifest) < 20:
        print(
            f"Warning: prepared {len(manifest)} clips. Official SA3 LoRA docs recommend at least 20-50 clips."
        )

    manifest_path = prepared_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} audio/caption pairs in {prepared_dir}")
    print(f"Manifest: {manifest_path}")
    return prepared_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Stable Audio 3 LoRA audio/caption pairs")
    parser.add_argument("--config", default="configs/cybermetal_lora_small_cpu.json")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_config(config_path)
    prepare(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
