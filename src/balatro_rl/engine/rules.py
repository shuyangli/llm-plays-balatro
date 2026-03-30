from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from balatro_rl.engine.models import Card, JokerInstance

CARD_CHIPS = {
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 10,
    12: 10,
    13: 10,
    14: 11,
}

HAND_BASES = {
    "high_card": (5, 1),
    "pair": (15, 2),
    "two_pair": (25, 2),
    "three_kind": (35, 3),
    "straight": (40, 4),
    "flush": (45, 4),
    "full_house": (50, 4),
    "four_kind": (70, 7),
    "straight_flush": (100, 8),
}


@dataclass(slots=True)
class ScoreResult:
    hand_name: str
    chips: int
    mult: int
    total_score: int
    joker_triggers: int
    triggered_jokers: list[str]
    updated_jokers: list[JokerInstance]


def _is_straight(ranks: list[int]) -> bool:
    if len(ranks) != 5:
        return False
    sorted_ranks = sorted(set(ranks))
    if len(sorted_ranks) != 5:
        return False
    if sorted_ranks[-1] - sorted_ranks[0] == 4:
        return True
    return sorted_ranks == [2, 3, 4, 5, 14]


def classify_hand(cards: Iterable[Card]) -> str:
    selected = list(cards)
    ranks = [card.rank for card in selected]
    suits = [card.suit for card in selected]
    counts = sorted({rank: ranks.count(rank) for rank in set(ranks)}.values(), reverse=True)
    is_flush = len(selected) == 5 and len(set(suits)) == 1
    is_straight = _is_straight(ranks)

    if is_straight and is_flush:
        return "straight_flush"
    if counts and counts[0] == 4:
        return "four_kind"
    if counts == [3, 2]:
        return "full_house"
    if is_flush:
        return "flush"
    if is_straight:
        return "straight"
    if counts and counts[0] == 3:
        return "three_kind"
    if counts.count(2) == 2:
        return "two_pair"
    if 2 in counts:
        return "pair"
    return "high_card"


def score_selected_hand(cards: list[Card], jokers: list[JokerInstance]) -> ScoreResult:
    hand_name = classify_hand(cards)
    base_chips, base_mult = HAND_BASES[hand_name]
    chips = base_chips + sum(CARD_CHIPS[card.rank] for card in cards)
    mult = base_mult
    joker_triggers = 0
    triggered_jokers: list[str] = []
    updated_jokers: list[JokerInstance] = []

    for joker in jokers:
        updated = joker
        if joker.kind == "chips":
            chips += joker.value
            joker_triggers += 1
            triggered_jokers.append(joker.name)
        elif joker.kind == "mult":
            mult += joker.value
            joker_triggers += 1
            triggered_jokers.append(joker.name)
        elif joker.kind == "xmult":
            mult *= joker.value
            joker_triggers += 1
            triggered_jokers.append(joker.name)
        elif joker.kind == "growth":
            bonus = joker.state.get("bonus_mult", 0)
            if hand_name != "high_card":
                bonus += joker.value
                updated = replace(joker, state={"bonus_mult": bonus})
            if bonus:
                mult += bonus
                joker_triggers += 1
                triggered_jokers.append(joker.name)
        updated_jokers.append(updated)

    total_score = chips * mult
    return ScoreResult(
        hand_name=hand_name,
        chips=chips,
        mult=mult,
        total_score=total_score,
        joker_triggers=joker_triggers,
        triggered_jokers=triggered_jokers,
        updated_jokers=updated_jokers,
    )


def blind_clear_money_bonus(jokers: list[JokerInstance]) -> tuple[int, int, list[str]]:
    total_bonus = 0
    triggers = 0
    names: list[str] = []
    for joker in jokers:
        if joker.kind == "economy":
            total_bonus += joker.value
            triggers += 1
            names.append(joker.name)
    return total_bonus, triggers, names
