# LLM RL Environment

- Owner: shuyangli
- Last updated: 2026-03-29
- Current status: complete for the first vertical slice; the Python package, deterministic engine core, reward registry, snapshots/replay, RL wrappers, validation tests, and a local human-play terminal UI are implemented. Immediate next action: expand content and rules toward closer Balatro parity without changing the environment contract.

## Motivation

We want a Balatro-like environment that can be controlled programmatically by LLMs and other RL agents. The core requirements are deterministic stepping, replayability, stable observations and action contracts, and reward shaping that is configurable without contaminating game logic.

## Design overview

The project will use a Python-first authoritative engine with no UI or network dependencies. The engine owns game state transitions, legal-action generation, scoring, shop progression, seeded randomness, and snapshot/replay support. Reward computation is a separate subsystem driven by deterministic transition metrics, with exactly one reward model active per episode.

The first implementation milestone is a vertical slice rather than full Balatro parity. It includes a blind selection phase, round play with poker-hand scoring, a shop with representative jokers, snapshot/restore, replay, built-in reward models, a Gymnasium-compatible wrapper, and tests that prove determinism.

## Details

The package is organized into engine modules:

- `core`: authoritative step logic and phase machine
- `rules`: hand evaluation and joker scoring effects
- `content`: declarative deck, blind, and shop data
- `rng`: deterministic substream RNG
- `serialization`: canonical snapshot serialization and hashing
- `metrics`: transition metrics emitted by the engine
- `rewards`: reward interface, registry, and built-in reward models
- `env`: RL-facing wrappers and episode runner

The canonical API exposes `EnvironmentConfig`, `Action`, `Observation`, `TransitionMetrics`, `RewardRegistry`, `RunSnapshot`, and `BalatroEnv`. The engine computes transition metrics first, then invokes the active reward model to obtain a scalar reward plus diagnostics.

For the vertical slice, blind start confirmation uses `noop` during the `blind_select` phase because the public action set intentionally stays small. That is an MVP compatibility choice and can later be replaced with a dedicated blind-start action if needed.

## Risks and mitigation

The main risk is overreaching toward full Balatro parity before the deterministic substrate is solid. Mitigation: prioritize exact replay behavior, canonical serialization, and modular content tables before expanding rule coverage.

Another risk is letting reward shaping leak into core logic. Mitigation: engine code emits metrics only; reward models are pure consumers of deterministic engine outputs and serialize any internal reward state explicitly.

## Milestones

1. Complete
Create the design doc and scaffold the Python package layout with the public interfaces.

2. Complete
Implement the deterministic engine, transition metrics, reward registry, snapshots, replay, and a minimal playable blind/hand/shop loop.

3. Complete
Add the RL wrapper, episode trace runner, and tests that cover determinism, snapshot/restore, reward behavior, invalid actions, and a minimal shop flow.

4. In progress
Expand content and rules toward closer Balatro parity, then harden the regression suite with more seeded fixtures and edge cases.

## Deployment and validation

Validation for the vertical slice is local and test-driven:

- run the unit test suite
- run a compile check over `src/` and `tests/`
- manually inspect a short episode trace for deterministic hashes and consistent reward outputs

Latest validation:

- `PYTHONPATH=src python3 -m unittest discover -s tests -v`
- `python3 -m compileall src tests`

No deployment surface exists yet because this milestone is a local library package only.
