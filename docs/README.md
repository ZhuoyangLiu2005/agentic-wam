# docs/

Start with the guide for what you are doing.

| doc | what it covers |
|---|---|
| [INSTALL.md](INSTALL.md) | environment setup and smoke tests |
| [PRETRAINING.md](PRETRAINING.md) | pretraining on the yam (robot) and ego (human-hand video) sources: data format, stats, text embeddings, code labels, configuration, launch |
| [DATA.md](DATA.md) | real-robot data preparation, end to end: recordings -> shards, state/action, stats, text embeddings, code labels, checks, the data config |
| [CODE_ANNOTATION.md](CODE_ANNOTATION.md) | policy-code labels for real-robot data — the VLM labeling pipeline or your own task program — and how they reach training |
| [REALWORLD.md](REALWORLD.md) | the real robots, YAM and Tianji + Sharpa: hardware, data, post-training, and running a checkpoint on YAM (inference server, robot contract, evaluation) |
| [ROBOTWIN.md](ROBOTWIN.md) | everything in RoboTwin simulation: code-labeled data collection (`codegen/sim`), training data and stage slicing, training, WAM evaluation, the hybrid code × WAM evaluation and its rollout viewer, model variants |

Module-level docs live with their code: `codegen/sim/{README,SETUP,CLAUDE}.md` (the simulation collection engine),
`codegen/pretrain/pipeline/` (the labeling engine; its config templates are `configs/{yam,ego}.yaml`),
`utils/yam_fk_ik/REPLAY_GUIDE.md` (YAM forward/inverse kinematics).

