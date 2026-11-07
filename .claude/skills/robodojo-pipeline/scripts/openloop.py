"""Open-loop inference of a RoboDojo checkpoint on its own training episodes. NOTHING MOVES.

Recorded 224x320 frames + raw 82-D state of the converted v2.1 shard -> model every `replan` frames ->
code + 64-D delta chunk. The chunk is resolved against the recorded state (YamUnifiedEEFTransform.backward,
the training inverse) and compared with the recorded action in the same unified EEF space: per row,
EEF position error (mm), geodesic rotation error (deg), |gripper error|. Mirrors scripts/sharpa_heldout_infer.py
(same loop, same outputs), with the policy built as deploy/robodojo/AgenticWAM does (task=posttrain_yam).

    CUDA_VISIBLE_DEVICES=<gpu> ~/venvs/agenticwam-data/bin/python openloop.py \
        --ckpt checkpoints/robodojo-stack-bowls/step20000/model.pt --episodes 0 --out runs/...
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import sharpa_heldout_infer as sh  # noqa: E402  (load_episode, install_capture, video_to_uint8, summarize)

logger = logging.getLogger("robodojo_openloop")
L_EEF, R_EEF, L_GRIP, R_GRIP = slice(0, 9), slice(32, 41), 31, 63


def build_policy(ckpt: Path, dataset: Path, overrides: list, nis: int, gen_dir: Path):
    sys.path.insert(0, str(REPO / "robotwin/deploy"))
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
    os.chdir(REPO)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    import deploy_policy as dp

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO / "configs")):
        cfg = compose(config_name="train", overrides=["task=posttrain_yam", f"data.dataset_root={dataset}"]
                      + list(overrides))
    OmegaConf.set_struct(cfg, False)
    tfs = [str(t.get("_target_", "")) for t in cfg.data.train.processor.get("action_state_transforms") or []]
    if not any(t.endswith("YamUnifiedEEFTransform") for t in tfs):
        raise SystemExit(f"posttrain_yam does not use YamUnifiedEEFTransform: {tfs}")
    nf = int(cfg.data.train.num_frames)
    fr = int(cfg.data.train.get("action_video_freq_ratio", 4))
    stats = dataset / "dataset_stats_eef64.json"
    policy = dp.AgenticWAMCosmosRobotWinPolicy(
        model_cfg=cfg.model, processor_cfg=cfg.data.train.processor, checkpoint_path=str(ckpt),
        dataset_stats_path=str(stats), device="cuda", action_horizon=nf - 1, replan_steps=24,
        num_inference_steps=nis, sigma_shift=None, seed=None, text_ckpt_dir=str(cfg.paths.text_encoder),
        text_max_len=128, n_cameras=int(cfg.model.get("n_cameras", 3)), num_video_frames=(nf - 1) // fr + 1,
        video_log_dir=str(gen_dir))
    if os.environ.get("AGENTICWAM_FAST_JOINT", "") not in ("", "0"):
        sys.path.insert(0, str(REPO / "deploy/common"))
        import fast_joint_infer
        fast_joint_infer.install(policy.model)
    return policy, cfg, str(stats)


class Yam64:
    def __init__(self):
        import torch
        from agenticwam.datasets.lerobot.transforms.yam_unified_eef import YamUnifiedEEFTransform, _rot6d_to_R
        self.torch, self.tf, self.rot = torch, YamUnifiedEEFTransform(keys=["default"]), _rot6d_to_R

    def abs64(self, raw82):
        return self.tf._state64(self.torch.as_tensor(raw82, dtype=self.torch.float64)).numpy()

    def pred_abs64(self, delta64, state82):
        t = self.torch
        st = self.tf._state64(t.as_tensor(state82, dtype=t.float64).reshape(1, 1, 82))
        batch = {"state": {"default": st}, "action": {"default": t.as_tensor(delta64, dtype=t.float64)[None]}}
        return self.tf.backward(batch)["action"]["default"][0].numpy()

    def errors(self, pred, gt):
        t = self.torch
        out = {}
        for side, eef, grip in (("L", L_EEF, L_GRIP), ("R", R_EEF, R_GRIP)):
            p, g = pred[:, eef], gt[:, eef]
            out[f"{side}_pos_mm"] = np.linalg.norm(p[:, :3] - g[:, :3], axis=-1) * 1e3
            Rp, Rg = self.rot(t.as_tensor(p[:, 3:9])), self.rot(t.as_tensor(g[:, 3:9]))
            cos = ((t.einsum("tji,tjk->tik", Rp, Rg).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
            out[f"{side}_rot_deg"] = t.rad2deg(t.arccos(cos)).numpy()
            out[f"{side}_grip_abs"] = np.abs(pred[:, grip] - gt[:, grip])
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True, help="training view dir")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--replan", type=int, default=24)
    ap.add_argument("--nis", type=int, default=6)
    ap.add_argument("--paths", default="r6000")
    ap.add_argument("--infer-code-tokens", type=int, default=128)
    ap.add_argument("--train-eps", type=Path, default=None, help="default <dataset>/episodes_0_94.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("AGENTICWAM_DEPLOY_DTYPE", "bf16")
    os.environ.setdefault("AGENTICWAM_FAST_JOINT", "1")
    os.environ.setdefault("AGENTICWAM_NOISE_SEED", "1234")
    os.environ["AGENTICWAM_SAVE_ALL_REPLANS"] = "1"
    os.environ.pop("AGENTICWAM_CODE_PIN", None)          # program regenerated every replan

    dataset = a.dataset.resolve()
    shard = dataset / "train_shard_0"
    instruction = json.loads((shard / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    trained = set(json.loads((a.train_eps or dataset / "episodes_0_94.json").read_text()))
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    overrides = [f"paths={a.paths}", f"model.infer_code_tokens={a.infer_code_tokens}"]
    t0 = time.perf_counter()
    policy, cfg, stats = build_policy(a.ckpt.resolve(), dataset, overrides, a.nis, out / "gen")
    last = sh.install_capture(policy.model)
    y64 = Yam64()
    H = policy.action_horizon
    (out / "session.json").write_text(json.dumps({
        "task": f"robodojo {instruction} (arx_x5)", "ckpt": str(a.ckpt), "dataset": str(dataset), "stats": stats,
        "episodes": a.episodes, "in_training_set": {e: e in trained for e in a.episodes}, "replan": a.replan,
        "nis": a.nis, "code_pin": False, "action_horizon": H, "overrides": overrides, "instruction": instruction,
        "build_sec": round(time.perf_counter() - t0, 1),
        "env": {k: v for k, v in os.environ.items() if k.startswith("AGENTICWAM_")}}, indent=2) + "\n")

    import torch
    summary = {}
    for episode in a.episodes:
        ep = sh.load_episode(shard, episode)
        n = ep["length"]
        ep_dir = out / f"ep_{episode:03d}"
        ep_dir.mkdir(exist_ok=True)
        policy.reset()
        policy.model._pinned_code_ids = None
        ts, deltas, preds, gts, videos, errs, recs = [], [], [], [], [], [], []
        t = 0
        with open(ep_dir / "codes.jsonl", "w") as codes_f:
            while t < n - 1:
                obs = {"observation": {cam: {"rgb": ep[cam][t]} for cam in sh.VIDEO_KEYS},
                       "joint_action": {"vector": ep["state"][t].tolist()}}
                ti = time.perf_counter()
                delta = np.asarray(policy._infer_action_chunk(obs, instruction), dtype=np.float64)
                infer_ms = (time.perf_counter() - ti) * 1e3
                policy.step_count += 1
                pred = y64.pred_abs64(delta, ep["state"][t])
                gt = np.full((H, 64), np.nan)
                k = min(H, n - t)
                gt[:k] = y64.abs64(ep["action"][t:t + k])
                e = y64.errors(pred, gt)
                ts.append(t); deltas.append(delta); preds.append(pred); gts.append(gt); errs.append(e)
                if last["video"] is not None:
                    videos.append(sh.video_to_uint8(last["video"], policy.n_cameras))
                rec = {"replan": len(ts) - 1, "t": t, "code": last["code"], "infer_ms": round(infer_ms, 1),
                       "exec": {kk: float(np.nanmean(v[:a.replan])) for kk, v in e.items()}}
                codes_f.write(json.dumps(rec) + "\n"); codes_f.flush()
                recs.append(rec)
                ex = rec["exec"]
                logger.info("ep %d replan %3d t=%4d/%d | %6.0f ms | pos mm L %.1f R %.1f | rot deg L %.1f R %.1f | "
                            "grip L %.3f R %.3f", episode, rec["replan"], t, n, infer_ms, ex["L_pos_mm"],
                            ex["R_pos_mm"], ex["L_rot_deg"], ex["R_rot_deg"], ex["L_grip_abs"], ex["R_grip_abs"])
                t += a.replan
        np.savez_compressed(ep_dir / "pred.npz", t=np.asarray(ts), delta64=np.stack(deltas),
                            pred_abs64=np.stack(preds), gt_abs64=np.stack(gts),
                            gen_video=np.stack(videos) if videos else np.zeros(0), length=n,
                            gt_abs64_full=y64.abs64(ep["action"]), state_abs64_full=y64.abs64(ep["state"]))
        codes = [r["code"] for r in recs]
        metrics = {"episode": episode, "in_training_set": episode in trained, "length": n, "n_replans": len(ts),
                   "exec_rows": sh.summarize(errs, slice(0, a.replan)), "all_rows": sh.summarize(errs, slice(0, H)),
                   "n_distinct_codes": len(set(codes)),
                   "infer_ms_p50": float(np.median([r["infer_ms"] for r in recs[1:] or recs])),
                   "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        (ep_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        summary[episode] = metrics
        logger.info("ep %d done: %s", episode, json.dumps(metrics["exec_rows"]))
    (out / "summary.json").write_text(json.dumps({"episodes": summary}, indent=2) + "\n")
    logger.info("summary: %s", out / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
