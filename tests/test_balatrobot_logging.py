from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from balatro_rl.balatrobot_openai import RunLogger, build_log_path


class BalatroBotLoggingTest(unittest.TestCase):
    def test_build_log_path_uses_jsonl_extension(self) -> None:
        path = build_log_path(Path("logs"))
        self.assertEqual(".jsonl", path.suffix)
        self.assertEqual(Path("logs"), path.parent)

    def test_run_logger_writes_jsonl_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "run.jsonl"
            logger = RunLogger(path)
            logger.log("model_choice", {"turn": 1, "name": "shop"})

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, len(lines))
            record = json.loads(lines[0])
            self.assertEqual("model_choice", record["event"])
            self.assertEqual(1, record["turn"])
            self.assertEqual("shop", record["name"])
            self.assertIn("timestamp", record)


if __name__ == "__main__":
    unittest.main()
