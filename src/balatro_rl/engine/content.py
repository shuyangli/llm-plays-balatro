from __future__ import annotations

from balatro_rl.engine.models import Blind, Card, JokerTemplate, ShopItem
from balatro_rl.engine.rng import DeterministicRNG

SUITS = ("spades", "hearts", "clubs", "diamonds")
RANKS = tuple(range(2, 15))
HAND_SIZE = 8
MAX_HAND_PLAYS = 4
MAX_DISCARDS = 4
MAX_JOKERS = 5
SHOP_SIZE = 2
BASE_REROLL_COST = 5
STARTING_MONEY = 10
MAX_ANTE = 2

BASE_BLIND_TARGETS = (300, 700, 1400)
BASE_BLIND_REWARDS = (4, 5, 6)
BOSS_NAMES = ("The Bot", "The Wall", "The Needle")

JOKER_LIBRARY: tuple[JokerTemplate, ...] = (
    JokerTemplate("chip_joker", "Chip Joker", "chips", 4, 30),
    JokerTemplate("mult_joker", "Mult Joker", "mult", 5, 4),
    JokerTemplate("xmult_joker", "XMult Joker", "xmult", 7, 2),
    JokerTemplate("economy_joker", "Economy Joker", "economy", 6, 2),
    JokerTemplate("growth_joker", "Growth Joker", "growth", 6, 1),
)


def build_standard_deck() -> list[Card]:
    deck: list[Card] = []
    for suit in SUITS:
        for rank in RANKS:
            deck.append(Card(id=f"{suit[:1]}-{rank:02d}", rank=rank, suit=suit))
    return deck


def get_blind(ante: int, index: int) -> Blind:
    scale = ante
    target = BASE_BLIND_TARGETS[index] * scale
    reward = BASE_BLIND_REWARDS[index] + (ante - 1)
    kind = ("small", "big", "boss")[index]
    boss_name = BOSS_NAMES[(ante + index - 1) % len(BOSS_NAMES)] if kind == "boss" else None
    return Blind(
        ante=ante,
        index=index,
        kind=kind,
        target_score=target,
        reward_money=reward,
        boss_name=boss_name,
    )


def generate_shop_inventory(
    shop_generation: int,
    rng: DeterministicRNG,
    size: int = SHOP_SIZE,
) -> list[ShopItem]:
    items: list[ShopItem] = []
    for slot in range(size):
        template = rng.choice(JOKER_LIBRARY, stream=f"shop-{shop_generation}")
        items.append(
            ShopItem(
                id=f"shop-{shop_generation}-{slot}",
                name=template.name,
                item_type="joker",
                price=template.cost,
                payload=template,
            )
        )
    return items
