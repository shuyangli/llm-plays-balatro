from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from balatro_rl.balatrobot_openai import BotConfig, RunLogger, build_log_path


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

    def test_run_logger_preserves_model_input_and_output_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "run.jsonl"
            logger = RunLogger(path)
            config = BotConfig(
                model="gpt-5-mini",
                deck="Red Deck",
                stake=1,
                seed=None,
                max_turns=10,
                port=12346,
                log_dir=Path(temp_dir),
            )

            logger.log(
                "model_input",
                {
                    "turn": 2,
                    "attempt": 1,
                    "state": "SHOP",
                    "model": config.model,
                    "input": [
                        {"role": "system", "content": "system prompt"},
                        {"role": "user", "content": '{"state":"SHOP"}'},
                    ],
                },
            )
            logger.log(
                "model_output",
                {
                    "turn": 2,
                    "attempt": 1,
                    "state": "SHOP",
                    "model": config.model,
                    "output": '{"name":"reroll","arguments":{}}',
                },
            )

            records = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(["model_input", "model_output"], [r["event"] for r in records])
            self.assertEqual("system", records[0]["input"][0]["role"])
            self.assertEqual(
                '{"name":"reroll","arguments":{}}',
                records[1]["output"],
            )


if __name__ == "__main__":
    unittest.main()
