# TensorZero Integration

- Owner: shuyangli
- Last updated: 2026-03-30
- Current status: complete for the first TensorZero integration slice; the bot now calls TensorZero over `/inference`, prompt text lives in TensorZero config files, and a Postgres-only local Docker Compose stack is checked in. Immediate next action: verify the compose stack against a live gateway and tighten any config details that differ across TensorZero releases.

## Motivation

The current bot embeds prompt construction directly in Python and calls the OpenAI Responses API without a gateway. That makes prompt iteration awkward, splits prompt logic across code and logs, and prevents us from using TensorZero's function configuration, observability, and experimentation workflow.

We want a local setup that can:

- run TensorZero locally with Docker Compose
- keep prompts and output schema in TensorZero config files instead of application code
- continue using JSONL run logs for bot-level tracing
- preserve the bot's current structured command contract

## Design overview

The bot will stop constructing full prompts in Python. Instead, Python will send structured template arguments to a TensorZero JSON function called `balatro_next_command`. TensorZero will own:

- the system prompt
- the user prompt template
- the output schema for the command payload
- the model selection for the default variant

Local development infrastructure will use Docker Compose with:

- `tensorzero/postgres` for Postgres-backed observability and metadata
- `tensorzero/gateway` for the inference API

The Python bot will call TensorZero through the OpenAI-compatible Responses API endpoint exposed by the gateway. Prompt variables will be passed using TensorZero template content blocks, and the returned JSON text will be validated against the existing game-state-specific command checks before the BalatroBot API call is executed.

## Details

Configuration layout:

- `config/tensorzero.toml` for the gateway function and variant configuration
- `config/functions/balatro_next_command/...` for prompt templates and the JSON output schema
- `docker-compose.yml` for local services

Inference flow:

1. Build a small structured payload from Python containing the current game state, allowed calls, run defaults, and the previous error.
2. Send that payload to TensorZero through the OpenAI SDK pointed at the gateway's `/openai/v1` endpoint, using a `tensorzero::function_name::...` model name and `tensorzero::template` content blocks.
3. Parse the returned JSON object from the response text.
4. Run the existing Python validation against the current game state before calling BalatroBot.

Logging:

- keep existing JSONL run logs in Python
- log the TensorZero request input payload and the TensorZero raw/parsed output
- do not duplicate prompt text generation in Python; the logged request should reflect template arguments rather than a flattened prompt string

Docker Compose:

- mount `./config` read-only into the gateway
- configure `TENSORZERO_POSTGRES_URL`
- require `OPENAI_API_KEY` in the gateway environment
- expose the gateway on `3000`

## Risks and mitigation

The main risk is API-shape drift between the TensorZero gateway and the currently pinned environment. Mitigation: keep the integration narrow, use the documented HTTP inference shape, and cover request construction plus response extraction with focused tests.

Another risk is duplicating prompt logic between Python and TensorZero. Mitigation: move the prompt body and JSON schema into TensorZero files and keep Python limited to passing structured arguments.

There is operational risk in relying on Postgres-only observability support while some TensorZero documentation still centers ClickHouse. Mitigation: keep the compose stack small, isolate TensorZero config under `config/`, and validate the checked-in setup against the currently deployed gateway image before depending on it for production workflows.

## Milestones

1. Complete
Create the design doc and decide the TensorZero integration shape.

2. Complete
Add TensorZero config, prompt templates, output schema, and Docker Compose services.

3. Complete
Refactor the Python bot to call TensorZero by function name and consume structured JSON output.

4. Complete
Add focused tests for request construction and structured-response handling, then validate locally.

## Deployment and validation

Deployment steps:

- export `OPENAI_API_KEY`
- run `docker compose up -d`
- run `docker compose run --rm tensorzero --run-postgres-migrations --config-file /app/config/tensorzero.toml`
- point the bot at the local gateway URL

Validation steps:

- run the focused Python unit tests
- verify the gateway loads `config/tensorzero.toml`
- confirm the bot logs TensorZero request and response records into the JSONL run log
- send a single manual inference through the local gateway and confirm a parsed JSON command comes back
