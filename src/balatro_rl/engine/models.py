from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ObservationMode = Literal["canonical", "tensor", "llm"]
IllegalActionMode = Literal["raise", "mask_only"]
ActionKind = Literal[
    "play_hand",
    "discard_cards",
    "apply_tarot",
    "buy_shop_item",
    "sell_joker",
    "reroll_shop",
    "move_joker",
    "choose_blind_reward",
    "skip_blind",
    "noop",
]
Phase = Literal["blind_select", "round_play", "shop", "terminal"]
TerminalOutcome = Literal["win", "loss", "in_progress"]


@dataclass(slots=True)
class EnvironmentConfig:
    seed: int = 1
    stake: int = 1
    deck_variant: str = "standard"
    content_profile: str = "default"
    reward_model: str = "terminal_win_loss"
    reward_params: dict[str, int | float | bool | str] = field(default_factory=dict)
    observation_mode: ObservationMode = "canonical"
    illegal_action_mode: IllegalActionMode = "raise"
    max_episode_steps: int | None = None


@dataclass(slots=True, frozen=True)
class Action:
    kind: ActionKind
    target_ids: tuple[str, ...] = ()
    target_slot: str | None = None
    metadata: tuple[tuple[str, str | int | bool], ...] = ()

    @classmethod
    def create(
        cls,
        kind: ActionKind,
        target_ids: list[str] | tuple[str, ...] | None = None,
        target_slot: str | None = None,
        metadata: dict[str, str | int | bool] | None = None,
    ) -> "Action":
        return cls(
            kind=kind,
            target_ids=tuple(sorted(target_ids or ())),
            target_slot=target_slot,
            metadata=tuple(sorted((metadata or {}).items())),
        )


@dataclass(slots=True)
class Card:
    id: str
    rank: int
    suit: str


@dataclass(slots=True)
class JokerTemplate:
    template_id: str
    name: str
    kind: Literal["chips", "mult", "xmult", "economy", "growth"]
    cost: int
    value: int


@dataclass(slots=True)
class JokerInstance:
    id: str
    template_id: str
    name: str
    kind: Literal["chips", "mult", "xmult", "economy", "growth"]
    cost: int
    value: int
    state: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ShopItem:
    id: str
    name: str
    item_type: Literal["joker"]
    price: int
    payload: JokerTemplate


@dataclass(slots=True)
class Blind:
    ante: int
    index: int
    kind: Literal["small", "big", "boss"]
    target_score: int
    reward_money: int
    boss_name: str | None = None


@dataclass(slots=True)
class TransitionMetrics:
    score_delta: int = 0
    money_delta: int = 0
    blind_cleared: bool = False
    blind_failed: bool = False
    ante_advanced: bool = False
    hands_spent: int = 0
    discards_spent: int = 0
    rerolls_spent: int = 0
    shop_purchase_count: int = 0
    joker_triggers: int = 0
    consumable_used: bool = False
    terminal_outcome: TerminalOutcome = "in_progress"
    snapshot_hash: str = ""


@dataclass(slots=True)
class Observation:
    run_id: str
    step_index: int
    phase: Phase
    rng_counters: dict[str, int]
    player_state: dict[str, Any]
    zones: dict[str, Any]
    blind_state: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    action_mask: list[bool]
    event_log_tail: list[str]
    derived_metrics: dict[str, Any]
    active_reward_model: str
    llm_view: dict[str, Any] | None = None


@dataclass(slots=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(slots=True)
class RunSnapshot:
    version: int
    config: dict[str, Any]
    state: dict[str, Any]
    rng_state: dict[str, Any]
    reward_state: dict[str, Any]
    snapshot_hash: str


@dataclass(slots=True)
class ReplayResult:
    rewards: list[float]
    snapshot_hashes: list[str]
    terminal_outcome: TerminalOutcome


@dataclass(slots=True)
class GameState:
    run_id: str
    phase: Phase
    step_index: int
    ante: int
    blind_index: int
    max_ante: int
    score: int
    score_target: int
    money: int
    hands_remaining: int
    discards_remaining: int
    reroll_cost: int
    draw_pile: list[Card]
    hand: list[Card]
    discard_pile: list[Card]
    played_cards: list[Card]
    jokers: list[JokerInstance]
    shop_inventory: list[ShopItem]
    event_log: list[str]
    shop_generation: int
    next_entity_id: int
    pending_blind_reward: int
    terminal_outcome: TerminalOutcome = "in_progress"


class InvalidActionError(ValueError):
    pass
