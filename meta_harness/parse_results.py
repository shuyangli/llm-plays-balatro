"""Parse JSONL game logs into structured metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_game_log(log_path: Path) -> dict[str, Any]:
    """Extract per-game metrics from a single JSONL run log."""
    events: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    metrics: dict[str, Any] = {
        "log_path": str(log_path),
        "ante_reached": 0,
        "run_finished_reason": None,
        "total_turns": 0,
        "error_count": 0,
        "model_error_count": 0,
        "api_error_count": 0,
    }

    for event in events:
        ev = event.get("event", "")
        if ev == "game_state":
            ante = _extract_ante(event.get("payload", {}))
            if ante is not None:
                metrics["ante_reached"] = max(metrics["ante_reached"], ante)
        elif ev == "run_finished":
            metrics["run_finished_reason"] = event.get("reason")
            metrics["total_turns"] = event.get("turn", 0)
        elif ev == "model_error":
            metrics["error_count"] += 1
            metrics["model_error_count"] += 1
        elif ev == "api_error":
            metrics["error_count"] += 1
            metrics["api_error_count"] += 1

    return metrics


def _extract_ante(game_state: dict[str, Any]) -> int | None:
    """Try multiple field paths to extract the current ante from a game state."""
    # Direct top-level "ante" field
    if isinstance(game_state.get("ante"), int):
        return game_state["ante"]

    # Nested under round_info / round / current_round
    for container_key in ("round_info", "round", "current_round"):
        container = game_state.get(container_key)
        if isinstance(container, dict):
            for ante_key in ("ante", "current_ante"):
                val = container.get(ante_key)
                if isinstance(val, int):
                    return val

    # Nested under game / G
    for container_key in ("game", "G"):
        container = game_state.get(container_key)
        if isinstance(container, dict):
            for ante_key in ("ante", "current_ante"):
                val = container.get(ante_key)
                if isinstance(val, int):
                    return val

    return None


def aggregate_scores(game_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of per-game metric dicts into summary statistics."""
    if not game_metrics:
        return {
            "n_games": 0,
            "avg_ante": 0.0,
            "max_ante": 0,
            "min_ante": 0,
            "error_rate": 0.0,
            "total_errors": 0,
            "reasons": [],
        }

    antes = [m["ante_reached"] for m in game_metrics]
    error_counts = [m["error_count"] for m in game_metrics]
    total_turns = sum(m["total_turns"] for m in game_metrics if m["total_turns"] > 0)

    return {
        "n_games": len(game_metrics),
        "avg_ante": sum(antes) / len(antes),
        "max_ante": max(antes),
        "min_ante": min(antes),
        "error_rate": sum(error_counts) / max(total_turns, 1),
        "total_errors": sum(error_counts),
        "reasons": [m["run_finished_reason"] for m in game_metrics],
    }
