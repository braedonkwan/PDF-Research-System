from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_llm.settings import load_settings


class SettingsHardCutoverTests(unittest.TestCase):
    def _load_settings_dict(self) -> dict:
        root = Path(__file__).resolve().parents[1]
        return json.loads((root / "settings.json").read_text(encoding="utf-8-sig"))

    def _write_temp_settings(self, payload: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp_path = Path(tmp.name)
        tmp.close()
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return tmp_path

    def test_new_runtime_schema_loads(self) -> None:
        payload = self._load_settings_dict()
        path = self._write_temp_settings(payload)
        try:
            config = load_settings(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(config.runtime.last_n_rounds, 1)
        self.assertFalse(config.runtime.debug_output)
        self.assertEqual(config.runtime.rag.top_k, 8)
        self.assertEqual(config.runtime.working_memory.recent_window_rounds, 5)
        self.assertEqual(config.runtime.working_memory.recent_top_k_rounds, 8)
        self.assertEqual(config.runtime.working_memory.older_top_k_rounds, 5)

    def test_runtime_debug_output_can_be_enabled(self) -> None:
        payload = self._load_settings_dict()
        payload.setdefault("runtime", {})["debug_output"] = True
        path = self._write_temp_settings(payload)
        try:
            config = load_settings(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertTrue(config.runtime.debug_output)

    def test_deprecated_rag_keys_fail_fast(self) -> None:
        payload = self._load_settings_dict()
        payload.setdefault("runtime", {}).setdefault("rag", {})["source_k"] = 3
        path = self._write_temp_settings(payload)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_settings(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertIn("Deprecated runtime.rag key(s)", str(ctx.exception))

    def test_deprecated_working_memory_keys_fail_fast(self) -> None:
        payload = self._load_settings_dict()
        payload.setdefault("runtime", {}).setdefault("working_memory", {})[
            "recent_turn_window"
        ] = 4
        path = self._write_temp_settings(payload)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_settings(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertIn("Deprecated runtime.working_memory key(s)", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
