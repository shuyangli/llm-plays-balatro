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
        parsed = parse_model_command('Action follows:\n{"name":"shop","arguments":{"action":"reroll"}}')
        self.assertEqual("shop", parsed["name"])
        self.assertEqual("reroll", parsed["arguments"]["action"])

    def test_normalize_state_name_accepts_integer_state(self) -> None:
        self.assertEqual("SHOP", normalize_state_name({"state": 5}))

    def test_allowed_calls_for_blind_select_state(self) -> None:
        self.assertEqual(
            ("skip_or_select_blind", "sell_joker", "sell_consumable", "use_consumable"),
            allowed_calls_for_state({"state_enum": "BLIND_SELECT"}),
        )

    def test_validate_play_hand_command(self) -> None:
        name, arguments = validate_command(
            {
                "name": "play_hand_or_discard",
                "arguments": {"action": "play_hand", "cards": [0, 2, 4]},
            },
            {"state_enum": "SELECTING_HAND"},
        )
        self.assertEqual("play_hand_or_discard", name)
        self.assertEqual({"action": "play_hand", "cards": [0, 2, 4]}, arguments)

    def test_validate_shop_buy_requires_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "arguments.index"):
            validate_command(
                {"name": "shop", "arguments": {"action": "buy_card"}},
                {"state_enum": "SHOP"},
            )


if __name__ == "__main__":
    unittest.main()
