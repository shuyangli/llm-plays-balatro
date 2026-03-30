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

    def test_round_play_maps_sorted_indices_to_direct_play_action(self) -> None:
        ui = TerminalUI(BalatroEnv(EnvironmentConfig(seed=3)))
        ui.env.reset()
        ui.env.step(Action.create("noop"))

        sorted_hand = ui._sorted_hand_cards(ui.env.observe())
        resolution = ui.resolve_command("play 1")
        self.assertEqual(1, len(resolution.actions))
        self.assertEqual("play_hand", resolution.actions[0].kind)
        self.assertEqual((sorted_hand[0]["id"],), resolution.actions[0].target_ids)

    def test_shop_buy_uses_indexed_inventory(self) -> None:
        ui = TerminalUI(BalatroEnv(EnvironmentConfig(seed=5)))
        ui.env.reset()
        ui.env.step(Action.create("noop"))
        ui.env.engine.state.score_target = 10
        ui.env.step(next(action for action in ui.env.legal_actions() if action.kind == "play_hand" and len(action.target_ids) == 5))

        observation = ui.env.observe()
        self.assertEqual("shop", observation.phase)
        resolution = ui.resolve_command("buy 1")
        self.assertEqual(1, len(resolution.actions))
        self.assertEqual("buy_shop_item", resolution.actions[0].kind)
        self.assertEqual(observation.zones["shop_inventory"][0]["id"], resolution.actions[0].target_ids[0])

    def test_render_sorts_hand_by_rank_then_suit(self) -> None:
        ui = TerminalUI(BalatroEnv(EnvironmentConfig(seed=3)))
        ui.env.reset()
        ui.env.step(Action.create("noop"))
        sorted_hand = ui._sorted_hand_cards(ui.env.observe())
        rendered = ui.render(ui.env.observe())
        first_card = f"1:{ui._format_card(sorted_hand[0])}"
        second_card = f"2:{ui._format_card(sorted_hand[1])}"
        self.assertIn(first_card, rendered)
        self.assertIn(second_card, rendered)


if __name__ == "__main__":
    unittest.main()
