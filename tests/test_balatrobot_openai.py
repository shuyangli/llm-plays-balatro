from __future__ import annotations

import unittest
from pathlib import Path

from balatro_rl.balatrobot_openai import (
    BotConfig,
    allowed_calls_for_state,
    build_model_input,
    normalize_state_name,
    parse_model_command,
    validate_command,
)


class BalatroBotOpenAITest(unittest.TestCase):
    def test_build_model_input_includes_system_and_user_messages(self) -> None:
        config = BotConfig(
            model="gpt-5-mini",
            deck="Red Deck",
            stake="WHITE",
            seed="seed-123",
            max_turns=10,
            port=12346,
            log_dir=Path("logs"),
        )
        model_input = build_model_input({"state": "SHOP"}, config, "bad output")

        self.assertEqual(2, len(model_input))
        self.assertEqual("system", model_input[0]["role"])
        self.assertEqual("user", model_input[1]["role"])
        self.assertIn('"previous_error": "bad output"', model_input[1]["content"])

    def test_parse_model_command_extracts_json_from_wrapped_text(self) -> None:
        parsed = parse_model_command('Action follows:\n{"name":"reroll","arguments":{}}')
        self.assertEqual("reroll", parsed["name"])
        self.assertEqual({}, parsed["arguments"])

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
