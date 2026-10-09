---
name: posttrain-annotation
description: "Give a downstream real-robot dataset (YAM or Tianji + Sharpa, already in the SFT format) per-frame code labels for post-training: route A with the VLM labeling engine (sub-task splits, labeling against the frozen skill library, the flat cache in the training view's order) or route B with a hand-written task program, then the alignment checks. Paid VLM stages need the user's explicit go-ahead. RoboTwin labels come from codegen/sim instead (robotwin-collect)."
---

# Code annotation for post-training

Guide: [docs/CODE_ANNOTATION.md](../../../docs/CODE_ANNOTATION.md) (§1–§6). The dataset must already be in the SFT
format — shards, `dataset_stats_eef64.json`, `cosmos_text_embeds_cache/` ([docs/DATA.md](../../../docs/DATA.md) steps
1–4); if it is not, do that first. Wrappers: `scripts/posttrain_annotation_yam.sh` (engine embodiment `yam_sft`, `yam`
skill library) and `scripts/posttrain_annotation_sharpa.sh` (`tianji_sharpa`, `ego` library).

## Ask first

1. **Robot and dataset**: YAM or Tianji + Sharpa, and `<dataset>` (the training view).
2. **Route.** A (VLM pipeline) for most datasets; B (one canonical task program + a frame-to-stage script the user
   owns) for a narrow task with fixed structure — branches, counted repetitions, a state that must be stated exactly
   (CODE_ANNOTATION.md §1, §5).
3. **For route A**: the VLM endpoint and key (`vlm_url` in the engine config, `VLM_API_KEY` or the git-ignored
   `codegen/pretrain/pipeline/.vlm_api_key`), a tag, and a budget. Paid stages: `subtasks` (without `--dry-run`) and
   `label`. State the number of episodes/slices and get an explicit yes before each.
4. **The labeling view** (Sharpa always; YAM optionally a higher-resolution copy): the same episodes in the same
   order as the training view. Sharpa's needs per-finger tactile and wrist poses as named state slices and a `top`
   video stream (CODE_ANNOTATION.md §4.2).

## Route A

```bash
# YAM
export YAM_SFT_DATASET=<dataset or its labeling view>
bash scripts/posttrain_annotation_yam.sh subtasks --dry-run          # free: gripper splits only — show them to the user
bash scripts/posttrain_annotation_yam.sh subtasks --workers 8        # PAID: VLM names each span -> <shard>/subtasks.json
bash scripts/posttrain_annotation_yam.sh label --tag <tag> --parallel 5 --tasks-file units.txt   # PAID
bash scripts/posttrain_annotation_yam.sh progress --tag <tag>
bash scripts/posttrain_annotation_yam.sh cache --train-view <dataset> --task <task> --tag <tag> \
     --work <scratch> --out <dataset>/policy_code_idx_<name>.pt

# Tianji + Sharpa: the same stages through posttrain_annotation_sharpa.sh, with
export TIANJI_DATASET=<labeling view>        # subtasks split on tactile contact onset/release
```

`units.txt` lists `"<dataset dir>:<episode>"` lines. Label a few episodes first, read the programs
(`codegen/pretrain/pipeline/outputs/<emb>/...`), and only then scale. Do not modify `codegen/pretrain/skill_library/`
during labeling — it changes `prompt_version()` and invalidates stored labels.

Alternatives for the cache step: `split-caches` (one cache per split, ordered by instantiating the loader) and `merge`
(several shard datasets with their caches).

## Route B

Follow CODE_ANNOTATION.md §5: write the canonical program once with the task's real control flow and the embodiment's
API and skill names; decide stage boundaries from sensors and temporal order, never appearance; measure every state
the code states from the episode; make each slice self-contained (`# STAGE k/N <NAME> | state: ... | next: ...`); write
the flat cache (§3: `{"codes": [...], "code_id_by_idx": int32[n_frames]}`) in the training view's frame order plus
`episode_order.json`. Keep the vocabulary small: the same situation must produce byte-identical text.

## Checks (both routes)

```bash
bash scripts/posttrain_annotation_<yam|sharpa>.sh verify --config configs/real_post_train/data/<yam|sharpa>.yaml \
     --dataset-root <dataset> --order <dataset>/episode_order.json                 # cache order == loader order
bash scripts/posttrain_annotation_<yam|sharpa>.sh dump --dataset <dataset> --cache <cache.pt> --out <dump_dir>
DRY_RUN=1 bash scripts/train_real_<yam|sharpa>.sh <dataset> <dataset>/policy_code_idx_<name>.pt
```

Report to the user: frames with code vs empty, `matched_raw_rate` and `repaired_share` for route A (never
`matched_slice_rate` / `pin_rate`), a few dumped labels, and the longest label in tokens — it sets
`model.infer_code_tokens` for training and serving (default 64).

Optional extras: `skill-prompt` (a skill-conditioning block per instruction → `data.skill_prompt_map`) and `subgoal`
(a language-only ablation cache with the same frame ids).

## Done when

`verify` and the dry run pass and the user has seen sample labels. Hand off to `downstream-training` with the cache
path and the `model.infer_code_tokens` value. If the dataset is re-converted later, rebuild or re-check the cache
(`rebuild-check`).
