#!/usr/bin/env python
"""Open-loop inference on held-out Sharpa episodes -- model output only, NOTHING MOVES.

The tong-transfer checkpoint was post-trained on episodes 0-99 of the converted v2.1 dataset. This
feeds the RECORDED observations of other episodes to the model and saves what it produces at each
replan: the generated code, the 64-D action chunk and the imagined future video. Ground truth is the
recorded action in the same 64-D unified-EEF space (no IK, no robot, no server).

    t = 0, R, 2R, ...:  frames[t] + state76[t]  ->  model  ->  code, delta64 [32,64], imagined video

Inputs come straight from the v2.1 shard (the exact 224x320 frames and raw 76-D state the model was
trained on), so neither the Sharpa camera crop nor FK is needed. The model is built the way
deploy/common/policy_runtime.py builds it (same hydra compose, same AgenticWAMCosmosRobotWinPolicy),
without PolicyRuntime itself: its Sharpa bridge imports pinocchio and its resolution guard only
accepts raw 640x360 frames.

    CUDA_VISIBLE_DEVICES=4 ~/venvs/agenticwam-data/bin/python scripts/sharpa_heldout_infer.py \\
        --dataset data_local/trex_tong_transfer_aw76_v21 --episodes 100 --max-replans 3

Writes runs/heldout_openloop/<session>/ep_NNN/{pred.npz,codes.jsonl,metrics.json} and
<session>/{session.json,summary.json}. Render the videos with scripts/sharpa_heldout_video.py.
"""
import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO / "checkpoints/tong-transfer-sharpa/step20000/model.pt"
DEFAULT_DATASET = REPO / "data_local/trex_tong_transfer_aw76_v21"
DEFAULT_EPISODES = list(range(100, 200, 10))

# contract camera key (deploy_policy._SIM_CAMERA_KEYS order) -> v2.1 video key
VIDEO_KEYS = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}

# 64-D unified layout (tianji_sharpa_unified_eef.py): L eef 0:9, hand 9:31, grip 31 | R eef 32:41, hand 41:63, grip 63
L_EEF, L_HAND, R_EEF, R_HAND = slice(0, 9), slice(9, 31), slice(32, 41), slice(41, 63)

logger = logging.getLogger("heldout")


# ── data ──────────────────────────────────────────────────────────────────────
def read_frames(path: Path) -> np.ndarray:
    """Every frame of one per-episode mp4 -> [N,H,W,3] uint8 RGB."""
    import av
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])])


def load_episode(shard: Path, episode: int) -> dict:
    """One v2.1 episode: raw 76-D state/action + the three 224x320 camera streams."""
    import pandas as pd
    info = json.loads((shard / "meta/info.json").read_text())
    chunk = episode // int(info.get("chunks_size", 1000))
    d = pd.read_parquet(shard / info["data_path"].format(episode_chunk=chunk, episode_index=episode))
    d = d.sort_values("frame_index")
    ep = {"state": np.stack(d["observation.state"].to_numpy()).astype(np.float64),
          "action": np.stack(d["action"].to_numpy()).astype(np.float64),
          "index": d["index"].to_numpy().astype(np.int64)}
    n = len(d)
    for cam, key in VIDEO_KEYS.items():
        frames = read_frames(shard / info["video_path"].format(episode_chunk=chunk, video_key=key,
                                                               episode_index=episode))
        if len(frames) < n:
            raise RuntimeError(f"episode {episode} {key}: {len(frames)} frames < {n} rows")
        ep[cam] = frames[:n]
    ep["length"] = n
    return ep


# ── model ─────────────────────────────────────────────────────────────────────
def build_policy(ckpt: Path, stats_dir: Path, overrides: list, nis: int, gen_dir: Path):
    """deploy/common/policy_runtime.py:PolicyRuntime.__init__ (hydra compose + policy build), minus
    the Sharpa FK/IK bridge and the raw-frame resolution guard."""
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "robotwin/deploy"))
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
    os.chdir(REPO)  # hydra composes with project-relative paths

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    import deploy_policy as dp

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO / "configs")):
        cfg = compose(config_name="train", overrides=["task=posttrain_sharpa", f"data.dataset_root={stats_dir}"]
                      + list(overrides))
    OmegaConf.set_struct(cfg, False)
    if int(cfg.model.get("action_dim", 0)) != 64:
        raise SystemExit(f"expected a 64-D unified-EEF task, got model.action_dim={cfg.model.get('action_dim')}")

    nf = int(cfg.data.train.num_frames)
    fr = int(cfg.data.train.get("action_video_freq_ratio", 4))
    policy = dp.AgenticWAMCosmosRobotWinPolicy(
        model_cfg=cfg.model, processor_cfg=cfg.data.train.processor, checkpoint_path=str(ckpt),
        dataset_stats_path=str(cfg.data.train.pretrained_norm_stats), device="cuda",
        action_horizon=nf - 1, replan_steps=24, num_inference_steps=nis, sigma_shift=None, seed=None,
        text_ckpt_dir=str(cfg.paths.text_encoder), text_max_len=128,
        n_cameras=int(cfg.model.get("n_cameras", 3)), num_video_frames=(nf - 1) // fr + 1,
        video_log_dir=str(gen_dir))
    if os.environ.get("AGENTICWAM_FAST_JOINT", "") not in ("", "0"):
        sys.path.insert(0, str(REPO / "deploy/common"))
        import fast_joint_infer
        fast_joint_infer.install(policy.model)
    return policy, cfg


def install_capture(model) -> dict:
    """Keep each call's (code, imagined video) -- the PolicyRuntime._install_code_capture wrapper,
    extended to the video. Pure passthrough otherwise."""
    last = {"code": None, "video": None}
    original = model.infer_action_trimodal

    def _capture(*args, **kwargs):
        out = original(*args, **kwargs)
        last["code"] = str(out[1]) if isinstance(out, tuple) and len(out) >= 2 else None
        last["video"] = out[2] if isinstance(out, tuple) and len(out) >= 3 else None
        return out

    model.infer_action_trimodal = _capture
    return last


def video_to_uint8(gen_video, n_cameras: int) -> np.ndarray:
    """[1,3,nc*T,H,W] in [-1,1] -> [nc,T,H,W,3] uint8 (view-major, as _save_generated_video)."""
    v = gen_video[0].float().clamp(-1, 1)
    C, T, H, W = v.shape
    tpv = max(1, T // n_cameras)
    v = v[:, :n_cameras * tpv].reshape(C, n_cameras, tpv, H, W)
    return ((v + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 3, 4, 0).cpu().numpy()


# ── comparison (64-D, no IK) ──────────────────────────────────────────────────
class Unified64:
    def __init__(self):
        import torch
        from agenticwam.datasets.lerobot.transforms.tianji_sharpa_unified_eef import (
            TianjiSharpaUnifiedEEFTransform, _rot6d_to_R)
        self.torch, self.tf, self.rot = torch, TianjiSharpaUnifiedEEFTransform(keys=["default"]), _rot6d_to_R

    def abs64(self, raw76: np.ndarray) -> np.ndarray:
        """[...,76] absolute raw -> [...,64] absolute unified."""
        return self.tf._state64(self.torch.as_tensor(raw76, dtype=self.torch.float64)).numpy()

    def pred_abs64(self, delta64: np.ndarray, state76: np.ndarray) -> np.ndarray:
        """[T,64] delta chunk anchored at state76 -> [T,64] absolute (the training inverse)."""
        t = self.torch
        st = self.tf._state64(t.as_tensor(state76, dtype=t.float64).reshape(1, 1, 76))
        batch = {"state": {"default": st}, "action": {"default": t.as_tensor(delta64, dtype=t.float64)[None]}}
        return self.tf.backward(batch)["action"]["default"][0].numpy()

    def errors(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        """Per-row errors between two [T,64] absolute chunks."""
        t = self.torch
        out = {}
        for side, eef, hand in (("L", L_EEF, L_HAND), ("R", R_EEF, R_HAND)):
            p, g = pred[:, eef], gt[:, eef]
            out[f"{side}_pos_mm"] = np.linalg.norm(p[:, :3] - g[:, :3], axis=-1) * 1e3
            Rp, Rg = self.rot(t.as_tensor(p[:, 3:9])), self.rot(t.as_tensor(g[:, 3:9]))
            cos = ((t.einsum("tji,tjk->tik", Rp, Rg).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
            out[f"{side}_rot_deg"] = t.rad2deg(t.arccos(cos)).numpy()
            out[f"{side}_hand_rad"] = np.abs(pred[:, hand] - gt[:, hand]).mean(-1)
        return out


def summarize(errs: list, rows: slice) -> dict:
    """Mean of each per-row error over `rows` of every chunk."""
    keys = errs[0].keys() if errs else []
    return {k: float(np.nanmean(np.concatenate([e[k][rows] for e in errs]))) for k in keys}


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="converted v2.1 dataset root")
    ap.add_argument("--episodes", type=int, nargs="+", default=DEFAULT_EPISODES)
    ap.add_argument("--replan", type=int, default=24, help="frames between inferences (deployment: 24)")
    ap.add_argument("--max-replans", type=int, default=None, help="per episode (pilot)")
    ap.add_argument("--nis", type=int, default=6, help="denoise steps (deployment: 6)")
    ap.add_argument("--code-pin", action="store_true",
                    help="keep the step-0 program for the whole episode, as deployment does")
    ap.add_argument("--paths", default="r6000")
    ap.add_argument("--override", action="append", default=[], help="extra hydra override")
    ap.add_argument("--out", type=Path, default=None, help="default runs/heldout_openloop/<timestamp>")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # deployment settings (deploy/launch_server_sharpa.sh); imagined video on every replan
    os.environ.setdefault("AGENTICWAM_DEPLOY_DTYPE", "bf16")
    os.environ.setdefault("AGENTICWAM_FAST_JOINT", "1")
    os.environ.setdefault("AGENTICWAM_NOISE_SEED", "1234")
    os.environ["AGENTICWAM_SAVE_ALL_REPLANS"] = "1"
    os.environ.pop("AGENTICWAM_CODE_PIN", None)  # pinning is done here, explicitly, with --code-pin

    dataset = a.dataset.resolve()
    shard = dataset / "train_shard_0"
    instruction = (dataset / "instructions.txt").read_text().strip().splitlines()[0]
    out = (a.out or REPO / "runs/heldout_openloop" / f"{dt.datetime.now():%Y%m%d_%H%M%S}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    trained = set(json.loads((dataset / "episodes_0_99.json").read_text()))
    if trained & set(a.episodes):
        logger.warning("episodes %s were in the TRAINING set", sorted(trained & set(a.episodes)))

    overrides = [f"paths={a.paths}", "model.infer_code_tokens=128", *a.override]
    t0 = time.perf_counter()
    policy, cfg = build_policy(a.ckpt.resolve(), a.ckpt.resolve().parent, overrides, a.nis, out / "gen")
    build_sec = time.perf_counter() - t0
    last = install_capture(policy.model)
    u64 = Unified64()
    H = policy.action_horizon
    (out / "session.json").write_text(json.dumps({
        "ckpt": str(a.ckpt), "dataset": str(dataset), "episodes": a.episodes, "replan": a.replan,
        "max_replans": a.max_replans, "nis": a.nis, "code_pin": a.code_pin, "action_horizon": H,
        "overrides": overrides, "instruction": instruction, "build_sec": round(build_sec, 1),
        "env": {k: v for k, v in os.environ.items() if k.startswith("AGENTICWAM_")},
        "model_dtype": str(next(policy.model.parameters()).dtype),
    }, indent=2) + "\n")

    import torch
    summary = {}
    for episode in a.episodes:
        ep = load_episode(shard, episode)
        n = ep["length"]
        ep_dir = out / f"ep_{episode:03d}"
        ep_dir.mkdir(exist_ok=True)
        policy.reset()
        policy.model._pinned_code_ids = None
        ts, deltas, preds, gts, videos, errs, recs = [], [], [], [], [], [], []
        t = 0
        with open(ep_dir / "codes.jsonl", "w") as codes_f:
            while t < n - 1:
                if a.max_replans is not None and len(ts) >= a.max_replans:
                    break
                obs = {"observation": {cam: {"rgb": ep[cam][t]} for cam in VIDEO_KEYS},
                       "joint_action": {"vector": ep["state"][t].tolist()}}
                ti = time.perf_counter()
                delta = np.asarray(policy._infer_action_chunk(obs, instruction), dtype=np.float64)  # [H,64]
                infer_ms = (time.perf_counter() - ti) * 1e3
                policy.step_count += 1
                if a.code_pin and policy.model._pinned_code_ids is None:
                    policy.model._pinned_code_ids = getattr(policy.model, "_last_code_canvas", None)

                pred = u64.pred_abs64(delta, ep["state"][t])
                gt = np.full((H, 64), np.nan)
                k = min(H, n - t)
                gt[:k] = u64.abs64(ep["action"][t:t + k])
                e = u64.errors(pred, gt)
                if last["video"] is None:
                    raise RuntimeError("no imagined video returned (return_video was off?)")
                ts.append(t); deltas.append(delta); preds.append(pred); gts.append(gt); errs.append(e)
                videos.append(video_to_uint8(last["video"], policy.n_cameras))
                rec = {"replan": len(ts) - 1, "t": t, "code": last["code"], "infer_ms": round(infer_ms, 1),
                       "exec": {kk: float(np.nanmean(v[:a.replan])) for kk, v in e.items()},
                       "row0_pos_mm": {s: float(e[f"{s}_pos_mm"][0]) for s in "LR"}}
                codes_f.write(json.dumps(rec) + "\n")
                codes_f.flush()
                recs.append(rec)
                ex = rec["exec"]
                logger.info("ep %d replan %3d t=%5d/%d | %5.0f ms | pos mm L %.1f R %.1f | rot deg L %.1f R %.1f | "
                            "hand rad L %.3f R %.3f", episode, rec["replan"], t, n, infer_ms,
                            ex["L_pos_mm"], ex["R_pos_mm"], ex["L_rot_deg"], ex["R_rot_deg"],
                            ex["L_hand_rad"], ex["R_hand_rad"])
                t += a.replan

        np.savez_compressed(
            ep_dir / "pred.npz", t=np.asarray(ts), delta64=np.stack(deltas), pred_abs64=np.stack(preds),
            gt_abs64=np.stack(gts), gen_video=np.stack(videos), length=n,
            gt_abs64_full=u64.abs64(ep["action"]), state_abs64_full=u64.abs64(ep["state"]))
        codes = [r["code"] for r in recs]
        metrics = {
            "episode": episode, "length": n, "frames_covered": min(t, n), "n_replans": len(ts),
            "exec_rows": summarize(errs, slice(0, a.replan)), "all_rows": summarize(errs, slice(0, H)),
            "row0_pos_mm": {s: float(np.mean([r["row0_pos_mm"][s] for r in recs])) for s in "LR"},
            "n_distinct_codes": len(set(codes)), "n_code_changes": sum(c != p for c, p in zip(codes[1:], codes)),
            "infer_ms_p50": float(np.median([r["infer_ms"] for r in recs[1:] or recs])),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        (ep_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        summary[episode] = metrics
        logger.info("ep %d done: %s", episode, json.dumps(metrics["exec_rows"]))

    eps = list(summary.values())
    agg = {rows: {k: float(np.mean([m[rows][k] for m in eps])) for k in eps[0][rows]}
           for rows in ("exec_rows", "all_rows")} if eps else {}
    (out / "summary.json").write_text(json.dumps({"mean_over_episodes": agg, "episodes": summary}, indent=2) + "\n")
    logger.info("summary: %s", out / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
