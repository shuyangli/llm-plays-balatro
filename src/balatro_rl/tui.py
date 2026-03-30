from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from balatro_rl.engine.env import BalatroEnv
from balatro_rl.engine.models import Action, EnvironmentConfig, InvalidActionError, Observation

RANK_LABELS = {
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}
SUIT_LABELS = {
    "spades": "S",
    "hearts": "H",
    "clubs": "C",
    "diamonds": "D",
}


@dataclass(slots=True)
class CommandResolution:
    actions: list[Action] = field(default_factory=list)
    message: str | None = None
    should_quit: bool = False
    should_reset: bool = False
    show_legal: bool = False


class TerminalUI:
    def __init__(self, env: BalatroEnv) -> None:
        self.env = env

    def run(self) -> None:
        self.env.reset()
        message = "New run started. Type 'help' for commands."
        while True:
            observation = self.env.observe()
            print(self.render(observation))
            if message:
                print(message)
                print()
            if observation.phase == "terminal":
                print("Run complete. Use 'reset' to start over or 'quit' to exit.")
            raw = input("> ").strip()
            resolution = self.resolve_command(raw)
            message = resolution.message

            if resolution.show_legal:
                message = self.render_legal_actions()
                continue
            if resolution.should_quit:
                return
            if resolution.should_reset:
                self.env.reset()
                message = "Run reset."
                continue

            if not resolution.actions:
                continue

            try:
                last_result = None
                for action in resolution.actions:
                    last_result = self.env.step(action)
                if last_result is not None:
                    metrics = last_result.info["transition_metrics"]
                    message = (
                        f"Applied {len(resolution.actions)} action(s). Reward={last_result.reward:.2f}; "
                        f"score_delta={metrics['score_delta']}; money_delta={metrics['money_delta']}."
                    )
            except InvalidActionError as error:
                message = str(error)

    def resolve_command(self, raw: str) -> CommandResolution:
        command = raw.strip().lower()
        if not command:
            return CommandResolution(message="No command entered.")
        if command in {"quit", "q", "exit"}:
            return CommandResolution(should_quit=True)
        if command in {"reset", "r"}:
            return CommandResolution(should_reset=True)
        if command in {"help", "h", "?"}:
            return CommandResolution(message=self.help_text())
        if command in {"legal", "ls"}:
            return CommandResolution(show_legal=True)

        observation = self.env.observe()
        try:
            if observation.phase == "blind_select":
                return self._resolve_blind_command(command)
            if observation.phase == "round_play":
                return self._resolve_round_command(command)
            if observation.phase == "shop":
                return self._resolve_shop_command(command)
            return CommandResolution(message="The run is over. Use 'reset' or 'quit'.")
        except ValueError as error:
            return CommandResolution(message=str(error))

    def render(self, observation: Observation) -> str:
        player = observation.player_state
        blind = observation.blind_state
        lines = [
            "\033[2J\033[HBalatro RL TUI",
            f"Run {observation.run_id} | Step {observation.step_index} | Phase {observation.phase}",
            (
                f"Ante {player['ante']} blind {player['blind_index']} ({blind['kind']}) | "
                f"Score {player['score']}/{player['score_target']} | Money ${player['money']}"
            ),
            (
                f"Hands {player['hands_remaining']} | Discards {player['discards_remaining']} | "
                f"Reroll ${player['reroll_cost']} | Reward model {observation.active_reward_model}"
            ),
            "",
            self._render_hand(observation),
            self._render_jokers(observation),
            self._render_shop(observation),
            self._render_events(observation),
            "",
            self._render_phase_help(observation.phase),
        ]
        return "\n".join(line for line in lines if line is not None)

    def render_legal_actions(self) -> str:
        legal = self.env.legal_actions()
        if not legal:
            return "No legal actions."
        lines = ["Legal actions:"]
        for index, action in enumerate(legal, start=1):
            lines.append(f"{index}. {self._format_action(action)}")
        return "\n".join(lines)

    def help_text(self) -> str:
        return (
            "Global commands: help, legal, reset, quit\n"
            "Blind select: start, skip\n"
            "Round play: toggle <card>, select <card...>, deselect <card...>, play, discard\n"
            "Shop: buy <item>, sell <joker>, reroll, move <joker> <slot>, next\n"
            "Indices shown in the UI are 1-based."
        )

    def _resolve_blind_command(self, command: str) -> CommandResolution:
        if command in {"start", "play", "next", "noop"}:
            return self._single_action("noop")
        if command == "skip":
            return self._single_action("skip_blind")
        return CommandResolution(message="Blind phase commands: start, skip")

    def _resolve_round_command(self, command: str) -> CommandResolution:
        parts = command.split()
        verb = parts[0]
        if verb in {"play", "p"}:
            return self._single_action("play_hand")
        if verb in {"discard", "d"}:
            return self._single_action("discard_selected")
        if verb in {"toggle", "t"} and len(parts) >= 2:
            return self._toggle_cards(parts[1:])
        if verb == "select" and len(parts) >= 2:
            return self._set_selection(parts[1:], target_selected=True)
        if verb == "deselect" and len(parts) >= 2:
            return self._set_selection(parts[1:], target_selected=False)
        return CommandResolution(message="Round commands: toggle <card>, select <cards>, deselect <cards>, play, discard")

    def _resolve_shop_command(self, command: str) -> CommandResolution:
        parts = command.split()
        verb = parts[0]
        if verb in {"next", "leave", "noop"}:
            return self._single_action("noop")
        if verb == "reroll":
            return self._single_action("reroll_shop")
        if verb == "buy" and len(parts) == 2:
            return self._shop_action("buy_shop_item", parts[1])
        if verb == "sell" and len(parts) == 2:
            return self._joker_action("sell_joker", parts[1])
        if verb == "move" and len(parts) == 3:
            return self._move_joker(parts[1], parts[2])
        return CommandResolution(message="Shop commands: buy <item>, sell <joker>, move <joker> <slot>, reroll, next")

    def _single_action(self, kind: str) -> CommandResolution:
        action = next((item for item in self.env.legal_actions() if item.kind == kind), None)
        if action is None:
            return CommandResolution(message=f"No legal action available for {kind}.")
        return CommandResolution(actions=[action])

    def _toggle_cards(self, indices: list[str]) -> CommandResolution:
        observation = self.env.observe()
        selected = set(observation.zones["selected_card_ids"])
        actions: list[Action] = []
        for index in self._parse_indices(indices, len(observation.zones["hand"])):
            card = observation.zones["hand"][index]
            kind = "deselect_card" if card["id"] in selected else "select_card"
            action = self._match_card_action(kind, card["id"])
            if action is None:
                return CommandResolution(message=f"Card {index + 1} cannot be toggled right now.")
            actions.append(action)
            if kind == "select_card":
                selected.add(card["id"])
            else:
                selected.remove(card["id"])
        return CommandResolution(actions=actions)

    def _set_selection(self, indices: list[str], target_selected: bool) -> CommandResolution:
        observation = self.env.observe()
        selected = set(observation.zones["selected_card_ids"])
        actions: list[Action] = []
        for index in self._parse_indices(indices, len(observation.zones["hand"])):
            card = observation.zones["hand"][index]
            card_selected = card["id"] in selected
            if target_selected and not card_selected:
                action = self._match_card_action("select_card", card["id"])
                if action is None:
                    return CommandResolution(message=f"Card {index + 1} cannot be selected.")
                actions.append(action)
                selected.add(card["id"])
            if not target_selected and card_selected:
                action = self._match_card_action("deselect_card", card["id"])
                if action is None:
                    return CommandResolution(message=f"Card {index + 1} cannot be deselected.")
                actions.append(action)
                selected.remove(card["id"])
        if not actions:
            return CommandResolution(message="Selection already matches the requested state.")
        return CommandResolution(actions=actions)

    def _shop_action(self, kind: str, index_text: str) -> CommandResolution:
        observation = self.env.observe()
        index = self._parse_single_index(index_text, len(observation.zones["shop_inventory"]))
        item = observation.zones["shop_inventory"][index]
        action = next(
            (candidate for candidate in self.env.legal_actions() if candidate.kind == kind and candidate.target_ids == (item["id"],)),
            None,
        )
        if action is None:
            return CommandResolution(message=f"Item {index + 1} is not buyable right now.")
        return CommandResolution(actions=[action])

    def _joker_action(self, kind: str, index_text: str) -> CommandResolution:
        observation = self.env.observe()
        index = self._parse_single_index(index_text, len(observation.zones["jokers"]))
        joker = observation.zones["jokers"][index]
        action = next(
            (candidate for candidate in self.env.legal_actions() if candidate.kind == kind and candidate.target_ids == (joker["id"],)),
            None,
        )
        if action is None:
            return CommandResolution(message=f"Joker {index + 1} is not available for {kind}.")
        return CommandResolution(actions=[action])

    def _move_joker(self, from_text: str, to_text: str) -> CommandResolution:
        observation = self.env.observe()
        jokers = observation.zones["jokers"]
        source_index = self._parse_single_index(from_text, len(jokers))
        target_index = self._parse_single_index(to_text, len(jokers))
        joker = jokers[source_index]
        action = next(
            (
                candidate
                for candidate in self.env.legal_actions()
                if candidate.kind == "move_joker"
                and candidate.target_ids == (joker["id"],)
                and candidate.target_slot == str(target_index)
            ),
            None,
        )
        if action is None:
            return CommandResolution(message=f"Cannot move joker {source_index + 1} to slot {target_index + 1}.")
        return CommandResolution(actions=[action])

    def _match_card_action(self, kind: str, card_id: str) -> Action | None:
        return next(
            (candidate for candidate in self.env.legal_actions() if candidate.kind == kind and candidate.target_ids == (card_id,)),
            None,
        )

    @staticmethod
    def _parse_indices(index_texts: list[str], upper_bound: int) -> list[int]:
        return [TerminalUI._parse_single_index(text, upper_bound) for text in index_texts]

    @staticmethod
    def _parse_single_index(index_text: str, upper_bound: int) -> int:
        if upper_bound <= 0:
            raise ValueError("There are no indexed items available.")
        try:
            index = int(index_text) - 1
        except ValueError as error:
            raise ValueError(f"Invalid index: {index_text}") from error
        if index < 0 or index >= upper_bound:
            raise ValueError(f"Index out of range: {index_text}")
        return index

    def _render_hand(self, observation: Observation) -> str:
        selected = set(observation.zones["selected_card_ids"])
        pieces = []
        for index, card in enumerate(observation.zones["hand"], start=1):
            marker = "*" if card["id"] in selected else " "
            pieces.append(f"{index}:{marker}{self._format_card(card)}")
        return "Hand: " + (" ".join(pieces) if pieces else "(empty)")

    def _render_jokers(self, observation: Observation) -> str:
        jokers = observation.zones["jokers"]
        if not jokers:
            return "Jokers: (none)"
        parts = [
            f"{index}:{joker['name']} [{joker['kind']}] value={joker['value']}"
            for index, joker in enumerate(jokers, start=1)
        ]
        return "Jokers: " + " | ".join(parts)

    def _render_shop(self, observation: Observation) -> str:
        if observation.phase != "shop":
            return "Shop: (not active)"
        items = observation.zones["shop_inventory"]
        if not items:
            return "Shop: (empty)"
        parts = [
            f"{index}:{item['name']} ${item['price']} [{item['payload']['kind']}]"
            for index, item in enumerate(items, start=1)
        ]
        return "Shop: " + " | ".join(parts)

    @staticmethod
    def _render_events(observation: Observation) -> str:
        if not observation.event_log_tail:
            return "Events: (none)"
        return "Events:\n" + "\n".join(f"  - {event}" for event in observation.event_log_tail)

    @staticmethod
    def _render_phase_help(phase: str) -> str:
        if phase == "blind_select":
            return "Commands: start, skip, legal, reset, quit"
        if phase == "round_play":
            return "Commands: toggle <card>, select <card...>, deselect <card...>, play, discard"
        if phase == "shop":
            return "Commands: buy <item>, sell <joker>, move <joker> <slot>, reroll, next"
        return "Commands: reset, quit"

    @staticmethod
    def _format_card(card: dict[str, object]) -> str:
        rank = int(card["rank"])
        rank_label = RANK_LABELS.get(rank, str(rank))
        return f"{rank_label}{SUIT_LABELS[str(card['suit'])]}"

    @staticmethod
    def _format_action(action: Action) -> str:
        target = ""
        if action.target_ids:
            target = f" targets={list(action.target_ids)}"
        if action.target_slot is not None:
            target += f" slot={action.target_slot}"
        return f"{action.kind}{target}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the Balatro RL environment from a terminal.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for deterministic runs.")
    parser.add_argument(
        "--reward-model",
        default="terminal_win_loss",
        help="Reward model id to use during play.",
    )
    args = parser.parse_args()

    config = EnvironmentConfig(
        seed=args.seed,
        reward_model=args.reward_model,
        observation_mode="canonical",
    )
    ui = TerminalUI(BalatroEnv(config))
    try:
        ui.run()
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
