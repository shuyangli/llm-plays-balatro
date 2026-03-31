from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from balatro_rl.balatrobot_openai import (
    ALL_BALATROBOT_ACTIONS,
    BotConfig,
    SUPPORTED_ACTION_SCHEMAS,
    allowed_calls_for_state,
    build_inference_input,
    extract_tensorzero_output,
    normalize_state_name,
    request_tensorzero_inference,
    run_bot,
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
        self.assertEqual(
            ALL_BALATROBOT_ACTIONS,
            tuple(json.loads(content["arguments"]["all_actions_json"])),
        )
        self.assertEqual(
            SUPPORTED_ACTION_SCHEMAS,
            json.loads(content["arguments"]["supported_action_schemas_json"]),
        )
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

    def test_allowed_calls_for_open_booster_state(self) -> None:
        self.assertEqual(
            ("pack",),
            allowed_calls_for_state({"state": "SMODS_BOOSTER_OPENED"}),
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

    def test_validate_buy_requires_one_shop_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "arguments.card"):
            validate_command(
                {"name": "buy", "arguments": {}},
                {"state": "SHOP"},
            )

    def test_validate_buy_accepts_card_target(self) -> None:
        name, arguments = validate_command(
            {"name": "buy", "arguments": {"card": 1}},
            {"state": "SHOP"},
        )
        self.assertEqual("buy", name)
        self.assertEqual({"card": 1}, arguments)

    def test_validate_pack_accepts_card_target(self) -> None:
        name, arguments = validate_command(
            {"name": "pack", "arguments": {"card": 0}},
            {"state": "SMODS_BOOSTER_OPENED"},
        )
        self.assertEqual("pack", name)
        self.assertEqual({"card": 0}, arguments)

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

    def test_request_tensorzero_inference_requests_raw_provider_payloads(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            def __init__(self) -> None:
                self.choices = [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": '{"name":"play","arguments":{"cards":[0]}}'})()},
                    )()
                ]

            def model_dump(self, mode: str = "json") -> dict[str, object]:
                testcase.assertEqual("json", mode)
                return {"id": "resp_123"}

        class FakeCompletions:
            def create(self, **kwargs: object) -> FakeResponse:
                captured.update(kwargs)
                return FakeResponse()

        class FakeChat:
            def __init__(self) -> None:
                self.completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, *, base_url: str, api_key: str) -> None:
                testcase.assertEqual("http://localhost:3000/openai/v1", base_url)
                testcase.assertEqual("tensorzero", api_key)
                self.chat = FakeChat()

        testcase = self
        with patch("openai.OpenAI", FakeOpenAI):
            raw_output, response_payload = request_tensorzero_inference(
                "http://localhost:3000",
                "balatro_next_command",
                [{"role": "user", "content": "hello"}],
                "episode-123",
            )

        self.assertEqual('{"name":"play","arguments":{"cards":[0]}}', raw_output)
        self.assertEqual({"id": "resp_123"}, response_payload)
        self.assertEqual(
            {
                "tensorzero::episode_id": "episode-123",
                "tensorzero::include_raw_response": True,
                "tensorzero::include_raw_usage": True,
            },
            captured["extra_body"],
        )

    def test_run_bot_continues_after_successful_first_attempt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = BotConfig(
                gateway_url="http://localhost:3000",
                function_name="balatro_next_command",
                deck="Red Deck",
                stake="WHITE",
                seed=None,
                max_turns=2,
                port=12346,
                log_dir=Path(temp_dir),
            )
            game_states = [
                {"state": "SELECTING_HAND"},
                {"state": "GAME_OVER"},
            ]
            api_results = [
                {"state": "ROUND_EVAL"},
            ]

            class FakeBalatroClient:
                def __init__(self, *, port: int) -> None:
                    self.port = port
                    self._game_states = list(game_states)
                    self._api_results = list(api_results)

                def call(self, name: str, arguments: dict | None = None) -> dict:
                    if name == "gamestate":
                        return self._game_states.pop(0)
                    self.assertEqual("play", name)
                    self.assertEqual({"cards": [0, 1]}, arguments)
                    return self._api_results.pop(0)

                def assertEqual(self, expected: object, actual: object) -> None:
                    testcase.assertEqual(expected, actual)

            testcase = self
            with patch("balatro_rl.balatrobot_openai.choose_command") as choose_command:
                choose_command.return_value = (
                    "play",
                    {"cards": [0, 1]},
                    '{"name":"play","arguments":{"cards":[0,1]}}',
                    "episode-123",
                )
                with patch("balatrobot.cli.client.BalatroClient", FakeBalatroClient):
                    run_bot(config)

            choose_command.assert_called_once()


if __name__ == "__main__":
    unittest.main()
