# Claude Code skills

Repository-wide skills for [Claude Code](https://claude.com/claude-code), in `skills/<name>/SKILL.md`. Claude Code
loads them when it is started in this repository; ask for a task in plain words ("set up this machine", "label my
YAM dataset", "launch post-training on …") or call a skill by name (`/agenticwam-setup`). Each skill is a runbook over
the scripts and guides in this repository: it asks for the inputs it needs, runs the steps in order, checks the gate
after each one, and stops for confirmation before anything that costs money (VLM calls) or occupies GPUs for long.

## Workflow skills

| Skill | What it does | Guide |
|---|---|---|
| [agenticwam-setup](skills/agenticwam-setup/SKILL.md) | environment, public weights + the AW-0 checkpoint, the paths file, environment variables, smoke tests | `docs/INSTALL.md` |
| [robotwin-dataprep](skills/robotwin-dataprep/SKILL.md) | collected `codegen/sim` datasets (or the official RoboTwin 2.0 set) → text-embedding cache, data + task configs, smoke test | `docs/ROBOTWIN.md` |
| [pretrain-annotation](skills/pretrain-annotation/SKILL.md) | per-frame code labels for the `yam` / `ego` pretraining sources with the VLM labeling engine, the flat cache, its checks | `docs/PRETRAINING.md` §2, `docs/CODE_ANNOTATION.md` §4 |
| [posttrain-annotation](skills/posttrain-annotation/SKILL.md) | per-frame code labels for a YAM or Tianji + Sharpa SFT dataset: the VLM pipeline or a task program, the flat cache, alignment checks | `docs/CODE_ANNOTATION.md` |
| [downstream-training](skills/downstream-training/SKILL.md) | real-robot post-training and RoboTwin training: dry-run gate, smoke, launch, monitoring, resume, then serving / benchmark | `docs/REALWORLD.md` §3, `docs/ROBOTWIN.md` |

A real-robot task goes `agenticwam-setup` → (dataset in the SFT format, `docs/DATA.md`) → `posttrain-annotation` →
`downstream-training`; a RoboTwin task goes `agenticwam-setup` → `robotwin-collect` → `robotwin-dataprep` →
`downstream-training`.

## RoboTwin program authoring (`codegen/sim`)

These drive `codegen/sim`: their agent rules are `codegen/sim/CLAUDE.md`, their commands run from `codegen/sim` after
`source env.sh`, and paths written in their text are relative to `codegen/sim`. The user-facing guide is
`docs/ROBOTWIN.md`

| Stage | Skill | What it does | Produces |
|---|---|---|---|
| Layer 2 — fix loop | [robotwin-fix-loop](skills/robotwin-fix-loop/) | coordinator + one Stage-1 subagent per task: diagnose baseline failures, author one program, then one-shot Stage-2 validation | `fix_code.py`, `findings.md`, a Stage-2 score, skills |
| Layer 3 — evosearch | [robotwin-evosearch](skills/robotwin-evosearch/) | K-candidate search for tasks the fix loop left below the bar for a strategy-level reason | `evosearch_best_code.py`, `findings.md` or `BLOCKED`, a Stage-2 score |
| Collection | [robotwin-collect](skills/robotwin-collect/) | frozen program → aligned collection → LeRobot dataset + slicing cache → `dataset.yaml` | `$CODEGEN_SIM_DATA_ROOT/<task>/` |
| Shared | [robotwin-program-authoring](skills/robotwin-program-authoring/) | observation-only robot-control knowledge, coordinator-owned | — |
| Shared | [program-contract.md](skills/robotwin-program-authoring/program-contract.md) | stage markers and slice shape every training program must carry | — |

Layer 1 (the LLM baseline that the fix loop diagnoses) has no runbook of its own:
`codegen/sim/iterate/scripts/run_robotwin_batch.py` and `docs/ROBOTWIN.md` cover it.

## Layout

```text
.claude/
  README.md                          this index
  skills/
    agenticwam-setup/                SKILL.md
    robotwin-dataprep/               SKILL.md
    pretrain-annotation/             SKILL.md
    posttrain-annotation/            SKILL.md
    downstream-training/             SKILL.md
    robotwin-program-authoring/      SKILL.md, program-contract.md, grasp/localize/transport/manipulation.md
    robotwin-fix-loop/               SKILL.md, INSTRUCTIONS.md, main-agent-prompt.md, subagent-prompt.md
    robotwin-evosearch/              ... + clean-task-slate.md
    robotwin-collect/                SKILL.md, INSTRUCTIONS.md
```

## Shared rules (canonical locations)

- Secrets (W&B, Hugging Face, VLM keys) come from the environment or a git-ignored file; no skill writes or prints them.
- RoboTwin seeds, forbidden access, runtime rules: [codegen/sim/CLAUDE.md](../codegen/sim/CLAUDE.md).
- Where RoboTwin artifacts land: [docs/ROBOTWIN.md](../docs/ROBOTWIN.md)
- Skill-library admission rules for code labeling: `codegen/pretrain/skill_library/{yam,ego}/CLAUDE.md`.
