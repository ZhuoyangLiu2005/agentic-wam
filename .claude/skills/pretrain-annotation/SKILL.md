---
name: pretrain-annotation
description: "Give the pretraining sources (yam robot data, ego human video) per-frame code labels with the VLM labeling engine in codegen/pretrain, via scripts/pretrain_annotation.sh: sub-task slices, Stage 1 labeling, skill mining, corpus-scale Stage 3, the flat cache in loader order, and its checks. Paid VLM stages need the user's explicit go-ahead."
---

# Code annotation for pretraining

Guides: [docs/PRETRAINING.md](../../../docs/PRETRAINING.md) §2 (the source must already have shards, stats and
instruction embeddings) and [docs/CODE_ANNOTATION.md](../../../docs/CODE_ANNOTATION.md) §4 (how the engine works).
Every stage goes through `bash scripts/pretrain_annotation.sh <stage> [args]`; its header lists what each stage runs
and which ones cost money.

## Ask first

1. **Which source** — `yam` or `ego` — and its root (`<yam_root>` / `<ego_root>`, standard LeRobot shards).
2. **The VLM endpoint**: `vlm_url` (any OpenAI-compatible chat-completions API) and the model name. The key goes in
   `VLM_API_KEY` or `codegen/pretrain/pipeline/.vlm_api_key` (chmod 600, git-ignored) — never in a config or a log.
3. **Budget.** Paid stages are `subtasks` (without `--dry-run`), `label` and `generate`. Before each, state the number
   of units / distinct labels and get an explicit yes. Never start a paid stage on your own initiative.
4. **Frozen or growing skill library?** The default is to use `codegen/pretrain/skill_library/{yam,ego}` as is.
   Changing a library changes `prompt_version()` and invalidates every stored label (paid regeneration).
5. **A tag** for this labeling run (e.g. `v1`); outputs are keyed by it.

## Steps

**0. Engine config.** Copy `codegen/pretrain/pipeline/configs/<source>.yaml` to a new file, set `yam_root` / `ego_root`,
`vlm_url` and the model, then `export CODEGEN_CONFIG=configs/<file>.yaml` (relative to `codegen/pretrain/pipeline`).
Install `codegen/pretrain/pipeline/requirements.txt` in the env that runs the engine (`PYTHON=<that python>`).

**1. Sub-task slices.** The engine labels one slice at a time, taking slices from an `annotation.sub_task` column, else
a per-shard `subtasks.json`, else one slice per episode. For yam without annotations:

```bash
bash scripts/pretrain_annotation.sh subtasks --dataset <yam_root> --dry-run     # free: inspect the gripper splits
bash scripts/pretrain_annotation.sh subtasks --dataset <yam_root> --workers 8   # PAID: VLM names each span
```

ego takes its events from `observation.hand_keypoints` when present; without it, only the annotation and frames
guide the labels (say so to the user).

**2. Work units.**

```bash
bash scripts/pretrain_annotation.sh progress --emit-frontier
cat codegen/pretrain/pipeline/outputs/<source>/frontier/shard_*.txt > units.txt     # "<dataset dir>:<episode>"
```

A subset is fine; start with ~50 units to check quality before scaling.

**3. Labels.** Small corpora (a few hundred episodes) can use Stage 1 directly; large ones use Stage 3, whose cost
scales with distinct sub-task labels:

```bash
# Stage 1 (PAID): the actor labels every unit
bash scripts/pretrain_annotation.sh label --embodiment <source> --tag <tag> --parallel 5 --tasks-file units.txt --skip-done
# Stage 3: one paid call per distinct label, broadcast to every episode
bash scripts/pretrain_annotation.sh inventory --embodiment <source> --units units.txt --out inv.json        # free
bash scripts/pretrain_annotation.sh generate  --embodiment <source> --inventory inv.json --parallel 5      # PAID
bash scripts/pretrain_annotation.sh assemble  --embodiment <source> --units units.txt --tag <tag>          # free
```

To grow the library instead: Stage 1 on a sample, `mine --embodiment <source> --tags <tag>`, admit skills against the
bar in `codegen/pretrain/skill_library/<source>/CLAUDE.md`, freeze, then label the corpus.

**4. Quality gate.** `bash scripts/pretrain_annotation.sh stats --embodiment <source>`. Report `matched_raw_rate` and
`repaired_share` — the only honest metrics. Never report `matched_slice_rate` or `pin_rate` as quality (a comment-only
program scores 1.0 on them). Read a handful of programs yourself under `codegen/pretrain/pipeline/outputs/`.

**5. Flat cache in loader order.**

```bash
bash scripts/pretrain_annotation.sh flatten --emb <source> --dataset-root <root> --tag <tag> --out <root>/policy_code_idx.pt
bash scripts/pretrain_annotation.sh verify --config configs/real_pre_train/data/pretrain_magic_soup.yaml \
     --source <source> --dataset-root <root> --order <root>/episode_order.json
```

`verify` must pass: a cache in the wrong frame order trains on the wrong labels with no error. Episodes without a
program get the empty code (masked out of the loss) — report how many.

## Done when

`<root>/policy_code_idx.pt` and `episode_order.json` exist, `verify` passes, and the user has the stats. Then:
`python scripts/launch_train.py --task pretrain_magic_soup --dry-run data.yam_root=<yam_root> data.ego_root=<ego_root>`
(PRETRAINING.md §4). Rebuild the cache (`flatten`) whenever the source's episodes change.
