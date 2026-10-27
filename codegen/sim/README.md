# RoboTwin programs to code-aligned training data

This workspace generates and validates Python robot-control programs, then replays them to collect
RoboTwin demonstrations with per-motion code provenance. The implementation lives in `capsim/`;
RoboTwin, model servers and the downstream trainer are external dependencies.

```text
Stage A: LLM baseline → Claude Code fix loop → optional evolutionary search
         → held-out validation → freeze.py → frozen/<task>/
Stage B: frozen/<task>/ → aligned collection → LeRobot export + stage slices → dataset.yaml
```

Stage A uses Claude Code as the coding agent. The baseline model uses a separate OpenAI-compatible
chat-completions endpoint. Candidate replay, validation and collection execute programs without LLM queries.

## Quick start

Follow this module's [setup guide](SETUP.md) for environment creation, dependency installation and
external assets. All setup files live here and can move with the module into a larger repository.

For later sessions, from `codegen/sim`:

```bash
conda activate codegen-sim
source env.sh
codegen_sim_preflight
```

- **Produce a program:** follow [Stage A commands](../../docs/ROBOTWIN.md),
  then [held-out validation and promotion](../../docs/ROBOTWIN.md).
- **Collect data:** with a validated handoff, follow [Stage B commands](../../docs/ROBOTWIN.md).
- **Work with Claude Code:** read [project rules](CLAUDE.md) and the [agent runbook index](../../.claude/README.md).

## Repository overview

| Path | Purpose |
|---|---|
| `iterate/` | Stage A scripts and replay configurations |
| `.claude/` | Agent runbooks, prompt templates, shared skills and program contract |
| `pipeline/` | Stage B collection and export drivers |
| `capsim/` | Shared simulation, robot APIs, tracing, model client and data conversion |
| `frozen/`, `outputs/` | Generated handoffs and run artifacts; git-ignored |

See [ROBOTWIN.md](../../docs/ROBOTWIN.md) for the scripts and what each stage hands over.

## Documentation

| Document | Responsibility |
|---|---|
| [Installation](SETUP.md) | Environment setup, external dependencies and readiness checks |
| [User guide](../../docs/ROBOTWIN.md) | Pipeline execution commands, recovery and troubleshooting |
| [CLAUDE.md](CLAUDE.md) | Agent constraints, runtime rules and seed policy |
| [Repository structure](../../docs/ROBOTWIN.md) | Source-file responsibilities and output locations |
| [Agent workflow](../../docs/ROBOTWIN.md) | Model/agent/replay roles and workflow design |
| [API tiers](../../docs/ROBOTWIN.md) | Robot functions and privilege semantics |
| [Program contract](../../.claude/skills/robotwin-program-authoring/program-contract.md) | Stage markers for code-stream training |
| [Handoff contract](../../docs/ROBOTWIN.md) | Frozen-program and dataset interfaces |

Before validation, read the [seed policy](CLAUDE.md#seed-policy). Before consuming generated data,
check the [handoff contract](../../docs/ROBOTWIN.md): a manifest alone does not establish training readiness.
