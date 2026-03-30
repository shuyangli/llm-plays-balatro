from __future__ import annotations

import copy
from itertools import combinations
from typing import Iterable

import balatro_rl.engine.content as content
import balatro_rl.engine.rules as rules
from balatro_rl.engine.metrics import metrics_to_dict
from balatro_rl.engine.models import (
    Action,
    EnvironmentConfig,
    GameState,
    InvalidActionError,
    JokerInstance,
    Observation,
    ReplayResult,
    StepResult,
    TransitionMetrics,
)
from balatro_rl.engine.rewards import RewardRegistry, build_default_registry
from balatro_rl.engine.rng import DeterministicRNG
from balatro_rl.engine.serialization import (
    action_to_dict,
    build_snapshot,
    environment_config_from_dict,
    game_state_from_dict,
    snapshot_hash,
)

MAX_EVENT_LOG = 20


class GameEngine:
    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        reward_registry: RewardRegistry | None = None,
    ) -> None:
        self.config = config or EnvironmentConfig()
        self.reward_registry = reward_registry or build_default_registry()
        self.reward_model = self.reward_registry.create(self.config.reward_model, self.config.reward_params)
        self.reward_state: dict[str, object] = {}
        self.rng = DeterministicRNG(self.config.seed)
        self.state = self._build_initial_state()
        self.reward_state = self.reward_model.reset(self.config, self.state)
        self.initial_snapshot = build_snapshot(
            self.config,
            self.state,
            self.rng.snapshot(),
            self.reward_state,
        )

    def reset(self, config_override: dict | None = None) -> tuple[Observation, dict[str, object]]:
        if config_override:
            next_config = copy.deepcopy(self.config)
            for key, value in config_override.items():
                setattr(next_config, key, value)
            self.config = next_config
        self.reward_model = self.reward_registry.create(self.config.reward_model, self.config.reward_params)
        self.rng = DeterministicRNG(self.config.seed)
        self.state = self._build_initial_state()
        self.reward_state = self.reward_model.reset(self.config, self.state)
        self.initial_snapshot = build_snapshot(
            self.config,
            self.state,
            self.rng.snapshot(),
            self.reward_state,
        )
        observation = self.observe()
        return observation, {"snapshot_hash": self.initial_snapshot.snapshot_hash}

    def legal_actions(self) -> list[Action]:
        phase = self.state.phase
        if phase == "terminal":
            return []
        if phase == "blind_select":
            actions = [Action.create("noop")]
            if self.state.blind_index < 2:
                actions.append(Action.create("skip_blind"))
            return actions
        if phase == "round_play":
            actions: list[Action] = []
            max_cards = min(5, len(self.state.hand))
            for count in range(1, max_cards + 1):
                for combo in combinations(self.state.hand, count):
                    card_ids = [card.id for card in combo]
                    actions.append(Action.create("play_hand", card_ids))
                    if self.state.discards_remaining > 0:
                        actions.append(Action.create("discard_cards", card_ids))
            return actions
        actions = [Action.create("noop")]
        for item in self.state.shop_inventory:
            if self.state.money >= item.price and len(self.state.jokers) < content.MAX_JOKERS:
                actions.append(Action.create("buy_shop_item", [item.id]))
        if self.state.money >= self.state.reroll_cost:
            actions.append(Action.create("reroll_shop"))
        for joker in self.state.jokers:
            actions.append(Action.create("sell_joker", [joker.id]))
        for index, joker in enumerate(self.state.jokers):
            for target_index in range(len(self.state.jokers)):
                if target_index != index:
                    actions.append(Action.create("move_joker", [joker.id], target_slot=str(target_index)))
        return actions

    def observe(self) -> Observation:
        blind = content.get_blind(self.state.ante, self.state.blind_index)
        legal = self.legal_actions()
        player_state = {
            "money": self.state.money,
            "hands_remaining": self.state.hands_remaining,
            "discards_remaining": self.state.discards_remaining,
            "score": self.state.score,
            "score_target": self.state.score_target,
            "ante": self.state.ante,
            "blind_index": self.state.blind_index,
            "reroll_cost": self.state.reroll_cost,
        }
        zones = {
            "draw_pile": [self._card_to_dict(card) for card in self.state.draw_pile],
            "hand": [self._card_to_dict(card) for card in self.state.hand],
            "discard_pile": [self._card_to_dict(card) for card in self.state.discard_pile],
            "played_cards": [self._card_to_dict(card) for card in self.state.played_cards],
            "jokers": [self._joker_to_dict(joker) for joker in self.state.jokers],
            "shop_inventory": [self._shop_item_to_dict(item) for item in self.state.shop_inventory],
            "consumables": [],
        }
        blind_state = {
            "kind": blind.kind,
            "ante": blind.ante,
            "target_score": blind.target_score,
            "reward_money": blind.reward_money,
            "boss_name": blind.boss_name,
        }
        derived_metrics = {
            "score_ratio": 0.0 if self.state.score_target == 0 else self.state.score / self.state.score_target,
            "deck_remaining": len(self.state.draw_pile) + len(self.state.discard_pile) + len(self.state.hand),
        }
        llm_view = None
        if self.config.observation_mode == "llm":
            llm_view = {
                "summary": (
                    f"Phase={self.state.phase}; money={self.state.money}; score={self.state.score}/"
                    f"{self.state.score_target}; hands={self.state.hands_remaining}; discards="
                    f"{self.state.discards_remaining}; jokers={[joker.name for joker in self.state.jokers]}"
                )
            }
        return Observation(
            run_id=self.state.run_id,
            step_index=self.state.step_index,
            phase=self.state.phase,
            rng_counters=dict(self.rng.counters),
            player_state=player_state,
            zones=zones,
            blind_state=blind_state,
            legal_actions=[action_to_dict(action) for action in legal],
            action_mask=[True] * len(legal),
            event_log_tail=self.state.event_log[-5:],
            derived_metrics=derived_metrics,
            active_reward_model=self.config.reward_model,
            llm_view=llm_view,
        )

    def snapshot(self):
        return build_snapshot(self.config, self.state, self.rng.snapshot(), self.reward_state)

    def restore(self, snapshot) -> None:
        self.config = environment_config_from_dict(snapshot.config)
        self.reward_model = self.reward_registry.create(self.config.reward_model, self.config.reward_params)
        self.state = game_state_from_dict(snapshot.state)
        self.rng = DeterministicRNG.restore(snapshot.rng_state)
        self.reward_state = dict(snapshot.reward_state)
        self.initial_snapshot = snapshot

    def replay(self, events: Iterable[Action]) -> ReplayResult:
        self.restore(self.initial_snapshot)
        rewards: list[float] = []
        hashes: list[str] = []
        for action in events:
            result = self.step(action)
            rewards.append(result.reward)
            hashes.append(result.info["snapshot_hash"])
            if result.terminated or result.truncated:
                break
        return ReplayResult(
            rewards=rewards,
            snapshot_hashes=hashes,
            terminal_outcome=self.state.terminal_outcome,
        )

    def step(self, action: Action) -> StepResult:
        legal = self.legal_actions()
        if action not in legal:
            return self._handle_invalid_action(action)

        prev_state = copy.deepcopy(self.state)
        metrics = TransitionMetrics()
        self.state.step_index += 1

        if action.kind == "noop":
            if self.state.phase == "blind_select":
                self._start_current_blind()
            elif self.state.phase == "shop":
                self._advance_from_shop(metrics)
        elif action.kind == "skip_blind":
            self.state.money += 1
            metrics.money_delta += 1
            self._append_event(f"Skipped {content.get_blind(self.state.ante, self.state.blind_index).kind} blind")
            self._advance_to_next_blind(metrics)
        elif action.kind == "play_hand":
            self._play_cards(action.target_ids, metrics)
        elif action.kind == "discard_cards":
            self._discard_cards(action.target_ids, metrics)
        elif action.kind == "buy_shop_item":
            self._buy_shop_item(action, metrics)
        elif action.kind == "sell_joker":
            self._sell_joker(action, metrics)
        elif action.kind == "reroll_shop":
            self._reroll_shop(metrics)
        elif action.kind == "move_joker":
            self._move_joker(action)
        else:
            raise InvalidActionError(f"Unsupported action kind: {action.kind}")

        reward, reward_info = self._apply_reward(prev_state, action, metrics)
        if self.config.max_episode_steps is not None and self.state.step_index >= self.config.max_episode_steps:
            self.state.phase = "terminal"
        if self.state.phase == "terminal" and metrics.terminal_outcome == "in_progress":
            metrics.terminal_outcome = self.state.terminal_outcome

        snapshot = self.snapshot()
        metrics.snapshot_hash = snapshot.snapshot_hash
        observation = self.observe()
        info = {
            "transition_metrics": metrics_to_dict(metrics),
            "reward_info": reward_info,
            "snapshot_hash": snapshot.snapshot_hash,
            "replay_cursor": self.state.step_index,
        }
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=self.state.phase == "terminal" and metrics.terminal_outcome in {"win", "loss"},
            truncated=self.config.max_episode_steps is not None and self.state.step_index >= self.config.max_episode_steps,
            info=info,
        )

    def _apply_reward(
        self,
        prev_state: GameState,
        action: Action,
        metrics: TransitionMetrics,
    ) -> tuple[float, dict[str, object]]:
        reward, next_reward_state, reward_info = self.reward_model.on_step(
            prev_state,
            action,
            metrics,
            self.state,
            self.reward_state,
        )
        self.reward_state = dict(next_reward_state)
        if self.state.phase == "terminal":
            terminal_reward, terminal_state, terminal_info = self.reward_model.on_terminal(
                self.state,
                metrics,
                self.reward_state,
            )
            reward += terminal_reward
            self.reward_state = dict(terminal_state)
            reward_info = {**reward_info, **terminal_info}
        return reward, reward_info

    def _handle_invalid_action(self, action: Action) -> StepResult:
        if self.config.illegal_action_mode == "raise":
            raise InvalidActionError(f"Illegal action in phase {self.state.phase}: {action}")
        snapshot = self.snapshot()
        observation = self.observe()
        return StepResult(
            observation=observation,
            reward=0.0,
            terminated=self.state.phase == "terminal",
            truncated=False,
            info={
                "invalid_action": action_to_dict(action),
                "transition_metrics": metrics_to_dict(TransitionMetrics(snapshot_hash=snapshot.snapshot_hash)),
                "reward_info": {},
                "snapshot_hash": snapshot.snapshot_hash,
                "replay_cursor": self.state.step_index,
            },
        )

    def _build_initial_state(self) -> GameState:
        deck = self.rng.shuffle(content.build_standard_deck(), stream="initial-deck")
        score_target = content.get_blind(1, 0).target_score
        run_id = snapshot_hash(self.config, GameState(
            run_id="pending",
            phase="blind_select",
            step_index=0,
            ante=1,
            blind_index=0,
            max_ante=content.MAX_ANTE,
            score=0,
            score_target=score_target,
            money=content.STARTING_MONEY,
            hands_remaining=content.MAX_HAND_PLAYS,
            discards_remaining=content.MAX_DISCARDS,
            reroll_cost=content.BASE_REROLL_COST,
            draw_pile=[],
            hand=[],
            discard_pile=[],
            played_cards=[],
            jokers=[],
            shop_inventory=[],
            event_log=[],
            shop_generation=0,
            next_entity_id=1,
            pending_blind_reward=content.get_blind(1, 0).reward_money,
        ), self.rng.snapshot(), {})[:12]
        return GameState(
            run_id=f"run-{run_id}",
            phase="blind_select",
            step_index=0,
            ante=1,
            blind_index=0,
            max_ante=content.MAX_ANTE,
            score=0,
            score_target=score_target,
            money=content.STARTING_MONEY,
            hands_remaining=content.MAX_HAND_PLAYS,
            discards_remaining=content.MAX_DISCARDS,
            reroll_cost=content.BASE_REROLL_COST,
            draw_pile=deck,
            hand=[],
            discard_pile=[],
            played_cards=[],
            jokers=[],
            shop_inventory=[],
            event_log=["Run initialized"],
            shop_generation=0,
            next_entity_id=1,
            pending_blind_reward=content.get_blind(1, 0).reward_money,
        )

    def _start_current_blind(self) -> None:
        blind = content.get_blind(self.state.ante, self.state.blind_index)
        self.state.phase = "round_play"
        self.state.score = 0
        self.state.score_target = blind.target_score
        self.state.hands_remaining = content.MAX_HAND_PLAYS
        self.state.discards_remaining = content.MAX_DISCARDS
        self.state.played_cards = []
        self.state.pending_blind_reward = blind.reward_money
        self.state.shop_inventory = []
        self.state.reroll_cost = content.BASE_REROLL_COST
        self._draw_to_hand()
        self._append_event(f"Started {blind.kind} blind with target {blind.target_score}")

    def _play_cards(self, card_ids: tuple[str, ...], metrics: TransitionMetrics) -> None:
        selected_ids = set(card_ids)
        selected = [card for card in self.state.hand if card.id in selected_ids]
        score_result = rules.score_selected_hand(selected, self.state.jokers)
        self.state.jokers = score_result.updated_jokers
        self.state.score += score_result.total_score
        metrics.score_delta += score_result.total_score
        metrics.hands_spent = 1
        metrics.joker_triggers += score_result.joker_triggers
        self._append_event(
            f"Played {score_result.hand_name} with {len(selected)} card(s) for {score_result.total_score} score"
        )

        remaining_hand = [card for card in self.state.hand if card.id not in selected_ids]
        self.state.discard_pile.extend(selected)
        self.state.hand = remaining_hand
        self.state.played_cards = []
        self.state.hands_remaining -= 1
        self._draw_to_hand()

        if self.state.score >= self.state.score_target:
            metrics.blind_cleared = True
            blind_bonus, bonus_triggers, bonus_names = rules.blind_clear_money_bonus(self.state.jokers)
            payout = self.state.pending_blind_reward + blind_bonus
            self.state.money += payout
            metrics.money_delta += payout
            metrics.joker_triggers += bonus_triggers
            self._append_event(f"Cleared blind and gained {payout} money")
            for name in bonus_names:
                self._append_event(f"{name} triggered on blind clear")
            if self.state.ante == self.state.max_ante and self.state.blind_index == 2:
                self.state.phase = "terminal"
                self.state.terminal_outcome = "win"
                metrics.terminal_outcome = "win"
                self._append_event("Run won")
            else:
                self.state.phase = "shop"
                self._refresh_shop()
        elif self.state.hands_remaining <= 0:
            self.state.phase = "terminal"
            self.state.terminal_outcome = "loss"
            metrics.blind_failed = True
            metrics.terminal_outcome = "loss"
            self._append_event("Run lost")

    def _discard_cards(self, card_ids: tuple[str, ...], metrics: TransitionMetrics) -> None:
        selected_ids = set(card_ids)
        selected = [card for card in self.state.hand if card.id in selected_ids]
        self.state.hand = [card for card in self.state.hand if card.id not in selected_ids]
        self.state.discard_pile.extend(selected)
        self.state.discards_remaining -= 1
        metrics.discards_spent = 1
        self._draw_to_hand()
        self._append_event(f"Discarded {len(selected)} cards")

    def _buy_shop_item(self, action: Action, metrics: TransitionMetrics) -> None:
        item_id = action.target_ids[0]
        item = next(item for item in self.state.shop_inventory if item.id == item_id)
        self.state.money -= item.price
        metrics.money_delta -= item.price
        metrics.shop_purchase_count = 1
        self.state.shop_inventory = [candidate for candidate in self.state.shop_inventory if candidate.id != item_id]
        joker = JokerInstance(
            id=f"joker-{self.state.next_entity_id}",
            template_id=item.payload.template_id,
            name=item.payload.name,
            kind=item.payload.kind,
            cost=item.payload.cost,
            value=item.payload.value,
            state={},
        )
        self.state.next_entity_id += 1
        self.state.jokers.append(joker)
        self._append_event(f"Bought {joker.name}")

    def _sell_joker(self, action: Action, metrics: TransitionMetrics) -> None:
        joker_id = action.target_ids[0]
        joker = next(joker for joker in self.state.jokers if joker.id == joker_id)
        sell_value = max(1, joker.cost // 2)
        self.state.jokers = [candidate for candidate in self.state.jokers if candidate.id != joker_id]
        self.state.money += sell_value
        metrics.money_delta += sell_value
        self._append_event(f"Sold {joker.name} for {sell_value}")

    def _reroll_shop(self, metrics: TransitionMetrics) -> None:
        cost = self.state.reroll_cost
        self.state.money -= cost
        metrics.money_delta -= cost
        metrics.rerolls_spent = 1
        self.state.reroll_cost += 1
        self._refresh_shop()
        self._append_event(f"Rerolled shop for {cost}")

    def _move_joker(self, action: Action) -> None:
        joker_id = action.target_ids[0]
        target_index = int(action.target_slot or 0)
        current_index = next(index for index, joker in enumerate(self.state.jokers) if joker.id == joker_id)
        joker = self.state.jokers.pop(current_index)
        self.state.jokers.insert(target_index, joker)
        self._append_event(f"Moved {joker.name} to slot {target_index}")

    def _advance_from_shop(self, metrics: TransitionMetrics) -> None:
        self._append_event("Left shop")
        self._advance_to_next_blind(metrics)

    def _advance_to_next_blind(self, metrics: TransitionMetrics) -> None:
        self.state.played_cards = []
        self.state.shop_inventory = []
        self.state.score = 0
        if self.state.blind_index == 2:
            if self.state.ante >= self.state.max_ante:
                self.state.phase = "terminal"
                self.state.terminal_outcome = "win"
                metrics.terminal_outcome = "win"
                return
            self.state.ante += 1
            self.state.blind_index = 0
            metrics.ante_advanced = True
        else:
            self.state.blind_index += 1
        blind = content.get_blind(self.state.ante, self.state.blind_index)
        self.state.phase = "blind_select"
        self.state.score_target = blind.target_score
        self.state.pending_blind_reward = blind.reward_money
        self._append_event(f"Advanced to {blind.kind} blind")

    def _refresh_shop(self) -> None:
        self.state.shop_generation += 1
        self.state.shop_inventory = content.generate_shop_inventory(self.state.shop_generation, self.rng)

    def _draw_to_hand(self) -> None:
        while len(self.state.hand) < content.HAND_SIZE:
            if not self.state.draw_pile:
                if not self.state.discard_pile:
                    break
                self.state.draw_pile = self.rng.shuffle(
                    self.state.discard_pile,
                    stream=f"reshuffle-{self.state.ante}-{self.state.blind_index}-{self.state.step_index}",
                )
                self.state.discard_pile = []
                self._append_event("Reshuffled discard pile")
            self.state.hand.append(self.state.draw_pile.pop(0))

    def _append_event(self, message: str) -> None:
        self.state.event_log.append(message)
        self.state.event_log = self.state.event_log[-MAX_EVENT_LOG:]

    @staticmethod
    def _card_to_dict(card) -> dict[str, object]:
        return {"id": card.id, "rank": card.rank, "suit": card.suit}

    @staticmethod
    def _joker_to_dict(joker) -> dict[str, object]:
        return {
            "id": joker.id,
            "template_id": joker.template_id,
            "name": joker.name,
            "kind": joker.kind,
            "cost": joker.cost,
            "value": joker.value,
            "state": dict(joker.state),
        }

    @staticmethod
    def _shop_item_to_dict(item) -> dict[str, object]:
        return {
            "id": item.id,
            "name": item.name,
            "item_type": item.item_type,
            "price": item.price,
            "payload": {
                "template_id": item.payload.template_id,
                "name": item.payload.name,
                "kind": item.payload.kind,
                "cost": item.payload.cost,
                "value": item.payload.value,
            },
        }
