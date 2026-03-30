from __future__ import annotations

import unittest

from balatro_rl.engine.env import BalatroEnv
from balatro_rl.engine.models import Action, EnvironmentConfig
from balatro_rl.tui import TerminalUI


class TerminalUITest(unittest.TestCase):
    def test_blind_commands_map_to_existing_actions(self) -> None:
        ui = TerminalUI(BalatroEnv(EnvironmentConfig(seed=3)))
        ui.env.reset()

        start = ui.resolve_command("start")
        self.assertEqual([Action.create("noop")], start.actions)

        skip = ui.resolve_command("skip")
        self.assertEqual([Action.create("skip_blind")], skip.actions)

    def test_round_toggle_selects_expected_card(self) -> None:
        ui = TerminalUI(BalatroEnv(EnvironmentConfig(seed=3)))
        ui.env.reset()
        ui.env.step(Action.create("noop"))

        resolution = ui.resolve_command("toggle 1")
        self.assertEqual(1, len(resolution.actions))
        self.assertEqual("select_card", resolution.actions[0].kind)
        self.assertEqual(ui.env.observe().zones["hand"][0]["id"], resolution.actions[0].target_ids[0])

    def test_shop_buy_uses_indexed_inventory(self) -> None:
        ui = TerminalUI(BalatroEnv(EnvironmentConfig(seed=5)))
        ui.env.reset()
        ui.env.step(Action.create("noop"))
        ui.env.engine.state.score_target = 10
        for _ in range(5):
            ui.env.step(next(action for action in ui.env.legal_actions() if action.kind == "select_card"))
        ui.env.step(next(action for action in ui.env.legal_actions() if action.kind == "play_hand"))

        observation = ui.env.observe()
        self.assertEqual("shop", observation.phase)
        resolution = ui.resolve_command("buy 1")
        self.assertEqual(1, len(resolution.actions))
        self.assertEqual("buy_shop_item", resolution.actions[0].kind)
        self.assertEqual(observation.zones["shop_inventory"][0]["id"], resolution.actions[0].target_ids[0])


if __name__ == "__main__":
    unittest.main()
