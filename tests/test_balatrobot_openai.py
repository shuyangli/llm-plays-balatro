from __future__ import annotations

import unittest

from balatro_rl.balatrobot_openai import (
    allowed_calls_for_state,
    normalize_state_name,
    parse_model_command,
    validate_command,
)


class BalatroBotOpenAITest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
