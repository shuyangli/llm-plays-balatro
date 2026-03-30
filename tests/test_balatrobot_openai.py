from __future__ import annotations

import unittest
from pathlib import Path

from balatro_rl.balatrobot_openai import (
    BotConfig,
    allowed_calls_for_state,
    build_inference_input,
    extract_tensorzero_output,
    normalize_state_name,
    validate_command,
)


class BalatroBotOpenAITest(unittest.TestCase):
    def test_build_inference_input_uses_template_arguments(self) -> None:
        config = BotConfig(
            gateway_url="http://localhost:3000",
            function_name="balatro_next_command",
            deck="Red Deck",
            stake="WHITE",
            seed="seed-123",
            max_turns=10,
            port=12346,
            log_dir=Path("logs"),
        )
        inference_input = build_inference_input({"state": "SHOP"}, config, "bad output")

        self.assertEqual(1, len(inference_input))
        self.assertEqual("user", inference_input[0]["role"])
        content = inference_input[0]["content"][0]
        self.assertEqual("tensorzero::template", content["type"])
        self.assertEqual("turn_context", content["name"])
        self.assertEqual("SHOP", content["arguments"]["state_name"])
        self.assertEqual("bad output", content["arguments"]["previous_error"])
        self.assertIn('"deck": "RED"', content["arguments"]["run_defaults_json"])

    def test_extract_tensorzero_output_returns_raw_and_parsed_json(self) -> None:
        raw_output, parsed_output, response_payload = extract_tensorzero_output(
            '{"name":"reroll","arguments":{}}',
            {"id": "resp_123", "model": "tensorzero::function_name::balatro_next_command"},
        )
        self.assertEqual('{"name":"reroll","arguments":{}}', raw_output)
        self.assertEqual({"name": "reroll", "arguments": {}}, parsed_output)
        self.assertEqual("resp_123", response_payload["id"])

    def test_normalize_state_name_accepts_string_state(self) -> None:
        self.assertEqual("SHOP", normalize_state_name({"state": "SHOP"}))

    def test_allowed_calls_for_blind_select_state(self) -> None:
        self.assertEqual(
            ("select", "skip"),
            allowed_calls_for_state({"state": "BLIND_SELECT"}),
        )

    def test_validate_play_command(self) -> None:
        name, arguments = validate_command(
            {
                "name": "play",
                "arguments": {"cards": [0, 2, 4]},
            },
            {"state": "SELECTING_HAND"},
        )
        self.assertEqual("play", name)
        self.assertEqual({"cards": [0, 2, 4]}, arguments)

    def test_validate_buy_requires_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "arguments.index"):
            validate_command(
                {"name": "buy", "arguments": {}},
                {"state": "SHOP"},
            )

    def test_validate_start_normalizes_human_friendly_values(self) -> None:
        name, arguments = validate_command(
            {
                "name": "start",
                "arguments": {"deck": "Red Deck", "stake": "white", "seed": "ABC123"},
            },
            {"state": "MENU"},
        )
        self.assertEqual("start", name)
        self.assertEqual(
            {"deck": "RED", "stake": "WHITE", "seed": "ABC123"},
            arguments,
        )


if __name__ == "__main__":
    unittest.main()
