---
name: robotwin-dataprep
description: "Turn RoboTwin data into a launchable AgenticWAM training config: collected codegen/sim datasets (dataset.yaml manifests) through robotwin/dataprep/prepare.py — text-embedding cache, data and task configs, smoke test — or the official RoboTwin 2.0 dataset for robotwin_wam. Use after robotwin-collect, before downstream-training."
---

# RoboTwin data preparation

The guide is [docs/ROBOTWIN.md](../../../docs/ROBOTWIN.md) Collection itself (writing, validating, freezing a
program and collecting demonstrations) is the `robotwin-collect` skill and §2. Run everything here from the repository
root in the **AgenticWAM** environment (not `codegen-sim`), with `PYTHONPATH=<repo>/src` first.

## Ask first

1. **Which data?**
   * **Collected with `codegen/sim`** → route A: one `dataset.yaml` per task under
     `codegen/sim/outputs/data/<task>/` (or the user's `CODEGEN_SIM_DATA_ROOT`).
   * **The official RoboTwin 2.0 demonstrations** → route B (`robotwin_wam`).
2. **A name** for the new configs (route A): `configs/robotwin/{data,task}/<name>.yaml`. Refuse a name that already
   exists unless the user wants it overwritten.
3. **Train split or all episodes** (`--split train|all`, default `train`).
4. **Disk for the text cache**: ~25 MB per distinct instruction; ten paraphrases per episode is ~12 GB per 60-episode
   task. It must be a local or fast disk the training nodes can read.

## Route A — collected data

**1. Check the manifests.** For each `dataset.yaml`: `lerobot_dir` exists, `episodes.kept` > 0, and `slices` is not
`null` (a null `slices` means the program had no stage markers — go back to `robotwin-collect`; the program contract
is `.claude/skills/robotwin-program-authoring/program-contract.md`).

**2. Machine paths.** `prepare.py` reads `norm_stats`, `text_cache_root`, `text_encoder`, `code_model`,
`foundation_ckpt` and `code_vision_path` from flags, else from the `dataprep:` block of
`configs/robotwin/site.yaml` (template `site.example.yaml`). Make sure each one resolves to an existing path.
`foundation_ckpt` is the checkpoint training starts from (usually `paths.robotwin_hybrid_init_ckpt` or the AW-0
pretrained checkpoint).

**3. Prepare** (the text-cache step needs a GPU; confirm the cost in disk with the user first):

```bash
python robotwin/dataprep/prepare.py --datasets <data_root>/*/dataset.yaml --name <name> [--split all]
```

It writes the text cache, `configs/robotwin/data/<name>.yaml`, `configs/robotwin/task/<name>.yaml` (from
`robotwin/dataprep/task_template.yaml`, the `robotwin_hybrid` recipe) and runs the smoke test.
`--skip-text-cache` reuses an existing cache; `--no-task-config` writes only the data config.

**4. Gate.** The smoke test must fetch samples whose `policy_code` starts with `# [stage` and whose `prompt_embs` are
present. Re-run it alone with `python robotwin/dataprep/smoke.py configs/robotwin/data/<name>.yaml --n 2`. Then
`DRY_RUN=1 bash scripts/train_robotwin.sh <name>` must pass its preflight.

## Route B — the official dataset

The `robotwin_wam` recipe reads `paths.robotwin_official` (`robotwin2.0/` LeRobot set + `dataset_stats.json`) and
`paths.robotwin_official_labels` (code labels, prompts, text embeddings; layout in ROBOTWIN.md). Nothing is
generated here: check both paths exist in the paths file, then `DRY_RUN=1 bash scripts/train_robotwin.sh robotwin_wam`.
Rebuild only the text embeddings if they are missing (ROBOTWIN.md, `scripts/precompute_cosmos_text_embeds.py`).
The official code labels are frozen artifacts and cannot be regenerated from this repository.

## Rules

* Never hand-edit `slices_ep/` or `cycle_by_idx.pt`; regenerate them with `robotwin/dataprep/slicing.py build`
  (ROBOTWIN.md) or by re-collecting.
* A stage-slice cache belongs to the program it was cut from: after re-freezing a program, re-collect or rebuild.
* Do not commit generated `configs/robotwin/{data,task}/<name>.yaml` unless the user asks; they name local paths.

## Done when

The smoke test and the training dry run pass. Hand off to `downstream-training` with
`bash scripts/train_robotwin.sh <name>` (or `robotwin_wam`).
