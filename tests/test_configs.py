import copy
import json
import unittest
from pathlib import Path

from scripts.run_sa3_workflow import load_config, validate_config
from scripts.run_sa3_medium_colab import normalize_config, normalize_source

ROOT = Path(__file__).resolve().parents[1]


class ConfigContracts(unittest.TestCase):
    def test_small_music_preset_and_duration_guard(self):
        config = load_config(ROOT / "configs/cyberpunk_industrial.json")
        config["smoke"]["duration_seconds"] = 121
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_medium_outputs_are_unique_per_job(self):
        config = {"model": {"name": "medium"}, "generation": {
            "prompt": "Instrumental electronic music", "output_name": "song.flac"}}
        first, _ = normalize_config(config, "job-one")
        second, _ = normalize_config(config, "job-two")
        self.assertNotEqual(first["generation"]["output_name"], second["generation"]["output_name"])
        self.assertEqual(first["generation"]["duration_seconds"], 380)

    def test_medium_rejects_overlong_audio(self):
        with self.assertRaises(ValueError):
            normalize_config({"model": {"name": "medium"}, "generation": {
                "prompt": "Music", "duration_seconds": 381}}, "test")

    def test_queue_preserves_order_and_unique_output_names(self):
        item = {"model": {"name": "medium"}, "generation": {"prompt": "Music"}}
        normalized, sources, ids = normalize_source(
            {"queue_version": 1, "items": [item, copy.deepcopy(item)]}, "queue")
        names = [g["output_name"] for g in normalized["generations"]]
        self.assertEqual(len(set(names)), 2)
        self.assertEqual(len(sources), 2)

    def test_presets_are_valid_json(self):
        for path in (ROOT / "configs").glob("*.json"):
            with self.subTest(path=path.name):
                self.assertIsInstance(json.loads(path.read_text()), (dict, list))
