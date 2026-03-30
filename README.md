# Balatro OpenAI Bridge

This repo includes `balatrobot-openai`, a small OpenAI-driven client for the
[`balatrobot`](https://pypi.org/project/balatrobot/) JSON-RPC server.

## Prerequisites

- Python 3.13+
- `uv`
- Balatro installed
- `balatrobot` working on your machine
- `OPENAI_API_KEY` set in your shell

## How To Run

Start the BalatroBot server first. This launches Balatro with the BalatroBot mod
loaded and starts the JSON-RPC server on port `12346` by default.

```bash
uv run python -m balatrobot serve
```

In another terminal, run the OpenAI bridge:

```bash
uv run balatrobot-openai --port 12346 --deck "Red Deck" --stake white
```

## Notes

- `balatrobot-openai` does not launch Balatro by itself.
- The bridge writes a JSONL log for every run under `logs/`.
- Stake values should be strings like `white`, `red`, `green`, `black`, `blue`,
  `purple`, `orange`, or `gold`.
- Deck names can be human-friendly values like `"Red Deck"` or `"Plasma Deck"`.

## Useful Checks

Check that the server is listening:

```bash
lsof -nP -iTCP:12346 -sTCP:LISTEN
```

Check server health:

```bash
uv run python -m balatrobot api health '{}'
```

Inspect the current game state:

```bash
uv run python -m balatrobot api gamestate '{}'
```
