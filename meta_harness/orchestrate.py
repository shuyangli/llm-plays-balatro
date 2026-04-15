"""
Meta-harness optimization loop for llm-plays-balatro.

Each iteration:
  1. Find the best-scoring candidate so far.
  2. Call the proposer (GPT-5.4-nano) to generate improved templates + preprocessing.
  3. Register the new TensorZero variant and restart the gateway.
  4. Evaluate the candidate over fixed seeds and save scores.
  5. Check early-stopping conditions.

Usage (from project root):
  uv run python -m meta_harness.orchestrate \\
      --seeds "SEED1,SEED2,SEED3" \\
      --iterations 10

Re-run from checkpoint (reuse existing scores.json for seed candidate):
  uv run python -m meta_harness.orchestrate \\
      --seeds "SEED1,SEED2,SEED3" \\
      --iterations 10 \\
      --skip-eval-seed

Early stopping (defaults):
  --patience 3          Stop if no candidate in the last 3 improves by >= min-delta
  --min-delta 0.3       Minimum avg_ante improvement to count as progress
  --win-threshold 6.0   Stop immediately if avg_ante reaches this value
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).parent.parent
CANDIDATES_DIR = PROJECT_DIR / "meta_harness" / "candidates"

_STRATEGY_V1_DIR = (
    PROJECT_DIR
    / "config"
    / "functions"
    / "balatro_next_command"
    / "variants"
    / "strategy_v1"
    / "templates"
)

# The seed candidate reuses the existing strategy_v1 TensorZero variant.
_SEED_VARIANT = "strategy_v1"

_DEFAULT_PREPROCESSING = '''\
"""
Preprocessing hook: runs before template rendering each turn.
Add computed features to the returned dict — do not remove existing keys.
Exceptions are caught automatically; the original game_state is used as fallback.
"""


def preprocess(game_state: dict) -> dict:
    return game_state
'''


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

def _check_early_stop(
    all_avg_antes: list[float],
    *,
    patience: int,
    min_delta: float,
    win_threshold: float,
) -> tuple[bool, str]:
    """
    Called after every newly scored candidate (seed included at index 0).
    Returns (should_stop, human-readable reason).
    """
    if not all_avg_antes:
        return False, ""

    latest = all_avg_antes[-1]

    if latest >= win_threshold:
        return True, f"win threshold reached (avg_ante={latest:.2f} ≥ {win_threshold})"

    # Only apply patience once we have: seed + at least `patience` new candidates.
    if len(all_avg_antes) < patience + 1:
        return False, ""

    recent_best = max(all_avg_antes[-patience:])
    # Baseline: best avg_ante before the current patience window.
    baseline = max(all_avg_antes[:-patience])

    if recent_best - baseline < min_delta:
        return True, (
            f"no improvement ≥ {min_delta} in last {patience} candidates "
            f"(recent best {recent_best:.2f}, baseline {baseline:.2f})"
        )

    return False, ""


# ---------------------------------------------------------------------------
# Candidate management
# ---------------------------------------------------------------------------

def _init_seed_candidate() -> Path:
    """Bootstrap candidate 000 from strategy_v1 templates."""
    candidate_dir = CANDIDATES_DIR / "000"
    harness_dir = candidate_dir / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)

    for template in ("system.minijinja", "turn_context.minijinja"):
        dst = harness_dir / template
        if not dst.exists():
            shutil.copy2(_STRATEGY_V1_DIR / template, dst)

    # Seed with a pass-through preprocessing.py so the proposer always sees one.
    pp = harness_dir / "preprocessing.py"
    if not pp.exists():
        pp.write_text(_DEFAULT_PREPROCESSING, encoding="utf-8")

    return candidate_dir


def _find_best_candidate() -> tuple[str, dict[str, Any]]:
    """Return the (id, scores) of the candidate with the highest avg_ante."""
    best_id: str | None = None
    best_scores: dict[str, Any] = {}
    best_avg = -1.0

    for scores_path in sorted(CANDIDATES_DIR.glob("*/scores.json")):
        candidate_id = scores_path.parent.name
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        avg_ante = float(scores.get("avg_ante", 0.0))
        if avg_ante > best_avg:
            best_avg = avg_ante
            best_id = candidate_id
            best_scores = scores

    if best_id is None:
        raise RuntimeError("No scored candidates found — run evaluation on the seed first.")
    return best_id, best_scores


def _next_id() -> str:
    """Return the next zero-padded three-digit candidate ID."""
    existing = [
        d.name for d in sorted(CANDIDATES_DIR.iterdir())
        if d.is_dir() and d.name.isdigit()
    ]
    return f"{int(existing[-1]) + 1:03d}" if existing else "001"


def _collect_history() -> list[dict[str, Any]]:
    """Return {id, avg_ante, analysis} for all candidates that have an analysis.md."""
    history = []
    for analysis_path in sorted(CANDIDATES_DIR.glob("*/analysis.md")):
        candidate_id = analysis_path.parent.name
        scores_path = analysis_path.parent / "scores.json"
        avg_ante = None
        if scores_path.exists():
            avg_ante = json.loads(scores_path.read_text(encoding="utf-8")).get("avg_ante")
        history.append({
            "id": candidate_id,
            "avg_ante": avg_ante,
            "analysis": analysis_path.read_text(encoding="utf-8"),
        })
    return history


def _collect_all_avg_antes() -> list[float]:
    """Return avg_ante for every scored candidate in id order (000, 001, …)."""
    result = []
    for scores_path in sorted(CANDIDATES_DIR.glob("*/scores.json")):
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        result.append(float(scores.get("avg_ante", 0.0)))
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(
    seeds: list[str],
    iterations: int,
    *,
    proposer_model: str,
    port: int,
    gateway_url: str,
    skip_eval_seed: bool,
    patience: int,
    min_delta: float,
    win_threshold: float,
) -> None:
    from meta_harness.evaluate import evaluate_candidate, register_variant, restart_gateway
    from meta_harness.propose import propose_candidate

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1 — seed candidate (000 = strategy_v1)                         #
    # ------------------------------------------------------------------ #
    seed_dir = _init_seed_candidate()

    if (seed_dir / "scores.json").exists():
        s = json.loads((seed_dir / "scores.json").read_text(encoding="utf-8"))
        print(f"[orchestrate] Seed candidate 000 already scored: avg_ante={s.get('avg_ante', 0):.2f}")
    elif skip_eval_seed:
        print("[orchestrate] --skip-eval-seed: skipping seed evaluation (no scores.json).")
    else:
        print("\n[orchestrate] Evaluating seed candidate 000 (strategy_v1)…")
        evaluate_candidate(seed_dir, _SEED_VARIANT, seeds, port=port)

    # Check early stop on seed alone (e.g., already above win threshold).
    all_avg_antes = _collect_all_avg_antes()
    stop, reason = _check_early_stop(
        all_avg_antes, patience=patience, min_delta=min_delta, win_threshold=win_threshold
    )
    if stop:
        print(f"\n[orchestrate] Early stop after seed: {reason}")
        _print_summary()
        return

    # ------------------------------------------------------------------ #
    # Step 2 — propose → register → evaluate loop                         #
    # ------------------------------------------------------------------ #
    for iteration in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f"[orchestrate] Iteration {iteration}/{iterations}")
        print(f"{'='*60}")

        best_id, best_scores = _find_best_candidate()
        best_dir = CANDIDATES_DIR / best_id
        print(f"[orchestrate] Best so far: {best_id}  avg_ante={best_scores.get('avg_ante', 0):.2f}")

        # Propose -----------------------------------------------------------
        new_id = _next_id()
        print(f"[orchestrate] Calling proposer (model={proposer_model}) → candidate {new_id}…")
        history = _collect_history()
        new_system, new_turn_ctx, new_preprocessing, analysis = propose_candidate(
            best_dir,
            model=proposer_model,
            all_candidates=history or None,
        )
        print(f"[orchestrate] Analysis: {analysis}")

        # Persist -----------------------------------------------------------
        new_dir = CANDIDATES_DIR / new_id
        harness_dir = new_dir / "harness"
        harness_dir.mkdir(parents=True, exist_ok=True)
        (harness_dir / "system.minijinja").write_text(new_system, encoding="utf-8")
        (harness_dir / "turn_context.minijinja").write_text(new_turn_ctx, encoding="utf-8")
        (harness_dir / "preprocessing.py").write_text(new_preprocessing, encoding="utf-8")
        (new_dir / "analysis.md").write_text(analysis, encoding="utf-8")

        # Register + restart ------------------------------------------------
        variant_name = f"meta_{new_id}"
        print(f"[orchestrate] Registering variant {variant_name} in tensorzero.toml…")
        register_variant(new_id, harness_dir)
        print("[orchestrate] Restarting TensorZero gateway…")
        restart_gateway(gateway_url=gateway_url)

        # Evaluate ----------------------------------------------------------
        print(f"[orchestrate] Evaluating {variant_name} on {len(seeds)} seed(s)…")
        new_scores = evaluate_candidate(new_dir, variant_name, seeds, port=port)

        # Delta vs previous best --------------------------------------------
        prev_avg = best_scores.get("avg_ante", 0.0)
        new_avg = new_scores.get("avg_ante", 0.0)
        delta = new_avg - prev_avg
        symbol = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        print(
            f"\n[orchestrate] {symbol} Candidate {new_id}: "
            f"avg_ante={new_avg:.2f} (Δ{delta:+.2f} vs {best_id})"
        )

        # Early stopping check ----------------------------------------------
        all_avg_antes = _collect_all_avg_antes()
        stop, reason = _check_early_stop(
            all_avg_antes, patience=patience, min_delta=min_delta, win_threshold=win_threshold
        )
        if stop:
            print(f"\n[orchestrate] Early stop: {reason}")
            break

    _print_summary()


def _print_summary() -> None:
    print(f"\n{'='*60}")
    print("[orchestrate] All candidates:")
    for scores_path in sorted(CANDIDATES_DIR.glob("*/scores.json")):
        candidate_id = scores_path.parent.name
        s = json.loads(scores_path.read_text(encoding="utf-8"))
        print(
            f"  {candidate_id}: avg_ante={s.get('avg_ante', 0):.2f}  "
            f"max_ante={s.get('max_ante', 0)}"
        )
    try:
        best_id, best_scores = _find_best_candidate()
        print(f"\n[orchestrate] Best: {best_id}  avg_ante={best_scores.get('avg_ante', 0):.2f}")
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meta-harness optimization loop for llm-plays-balatro.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seeds",
        default="SEED1,SEED2,SEED3",
        help="Comma-separated Balatro seeds used for every evaluation.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Maximum number of propose-evaluate iterations.",
    )
    parser.add_argument(
        "--proposer-model",
        default="gpt-5.4-nano-2026-03-17",
        help="OpenAI model used by the proposer.",
    )
    parser.add_argument("--port", type=int, default=12346, help="BalatroBot TCP port.")
    parser.add_argument(
        "--gateway-url",
        default="http://localhost:3000",
        help="TensorZero gateway base URL.",
    )
    parser.add_argument(
        "--skip-eval-seed",
        action="store_true",
        help="Skip evaluation of candidate 000 (useful when resuming from a checkpoint).",
    )

    # Early-stopping knobs
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop if avg_ante hasn't improved by min-delta in this many consecutive candidates.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.3,
        help="Minimum avg_ante improvement over the patience window to continue.",
    )
    parser.add_argument(
        "--win-threshold",
        type=float,
        default=6.0,
        help="Stop immediately when avg_ante reaches this value.",
    )

    args = parser.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        parser.error("--seeds must contain at least one non-empty value.")

    run_loop(
        seeds,
        args.iterations,
        proposer_model=args.proposer_model,
        port=args.port,
        gateway_url=args.gateway_url,
        skip_eval_seed=args.skip_eval_seed,
        patience=args.patience,
        min_delta=args.min_delta,
        win_threshold=args.win_threshold,
    )


if __name__ == "__main__":
    main()
