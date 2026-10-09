#!/usr/bin/env python
"""AGENTICWAM-Cosmos real-world inference SERVER (ABC bottles / bimanual YAM, joint co-denoise).

Runs on the MODEL's env + GPU. Loads a trained agenticwam_cosmos checkpoint once, then serves obs->action
over HTTP: the robot client POSTs the current observation (3 camera RGBs + 14-D joint state) and gets
back an ABSOLUTE 14-D qpos action chunk.

Reuses the PROVEN obs->action pipeline in
  robotwin/deploy/deploy_policy.py :: AgenticWAMCosmosRobotWinPolicy
  (multiview obs [1,3,nc,224,320] in [-1,1]  ->  Cosmos-Reason1 prompt_embs  ->
   infer_action_trimodal -> _infer_action_joint co-denoise  ->  processor de-normalization),
but composed directly from a real-world hydra task config (NO robotwin sim / NO cuRobo IK), and
converts the model's RELATIVE-joint output back to ABSOLUTE qpos with the request's current state
(RelativeJointTransform.backward: qpos = relative_action + current_state).

Works for BOTH model variants — the policy branches on cfg.model.use_code_expert:
  --task <3-expert config>  -> video+code+action, infer_action_trimodal
  --task <2-expert config>  -> video+action,      infer_action

Endpoints:
  GET  /health  -> model + config info
  POST /reset   -> start a new episode (clears the per-episode code cache)
  POST /act     -> {instruction, images:{head_camera,left_camera,right_camera: b64-PNG}, state:[14]}
                   returns {action_chunk:[[..14]..T], horizon, replan_steps,
                            action_type:"qpos_abs", instruction, code}
                   The client executes the first `replan_steps` rows as absolute joint targets,
                   then calls /act again on a fresh observation (receding-horizon re-plan).

Usage:
  DIFFSYNTH_DOWNLOAD_SOURCE=huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    python deploy/deploy_server.py --ckpt runs/<run>/checkpoints/weights/step_100000.pt \
      --task posttrain_yam --dataset <dataset_dir> --instruction "put the plastic bottles in the bin" --port 8711
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# deploy/ lives directly under agenticwam_cosmos/, so parents[1] == project root
AGENTICWAM = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, AGENTICWAM + "/src")
sys.path.insert(0, AGENTICWAM + "/robotwin/deploy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained step_XXXXXX.pt (pass a nonexistent path for load_pretrain smoke mode)")
    ap.add_argument("--task", default="posttrain_yam", help="the training task config (configs/real_post_train/task/)")
    ap.add_argument("--dataset", default=None, help="the dataset directory the checkpoint was post-trained on (its stats + processing)")
    ap.add_argument("--override", action="append", default=[], help="extra hydra override, e.g. model.infer_code_tokens=192 (repeatable)")
    ap.add_argument("--dataset-stats", default=None, help="dataset_stats_deltajoint.json (default: cfg.data.train.pretrained_norm_stats)")
    ap.add_argument("--action-horizon", type=int, default=None, help="default: num_frames-1 (=32)")
    ap.add_argument("--replan-steps", type=int, default=8, help="rows the client executes before re-planning")
    ap.add_argument("--num-inference-steps", type=int, default=20, help="video/action denoise steps")
    ap.add_argument("--instruction", default="put the plastic bottles in the bin", help="fallback task instruction")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8711)
    a = ap.parse_args()

    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
    os.chdir(AGENTICWAM)  # hydra composes configs with project-relative paths

    import numpy as np
    import torch  # noqa: F401  (imported so a missing CUDA fails loudly at startup)
    from PIL import Image
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    import deploy_policy as dp  # robotwin/deploy/deploy_policy.py

    # ---- compose the ABC config (model + processor + data params); no robotwin sim ----
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=os.path.join(AGENTICWAM, "configs")):
        cfg = compose(config_name="train", overrides=[f"task={a.task}"] + ([f"data.dataset_root={a.dataset}"] if a.dataset else []) + a.override)
    OmegaConf.set_struct(cfg, False)

    ds = a.dataset_stats or str(cfg.data.train.pretrained_norm_stats)
    nf = int(cfg.data.train.num_frames)
    fr = int(cfg.data.train.get("action_video_freq_ratio", 4))
    horizon = int(a.action_horizon) if a.action_horizon else (nf - 1)
    num_video_frames = (nf - 1) // fr + 1
    n_cameras = int(cfg.model.get("n_cameras", 3))

    print(f"[server] building policy | ckpt={a.ckpt} | task={a.task} "
          f"| code_expert={bool(cfg.model.get('use_code_expert', False))} "
          f"| horizon={horizon} replan={a.replan_steps} nvf={num_video_frames}", flush=True)
    policy = dp.AgenticWAMCosmosRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=a.ckpt,
        dataset_stats_path=ds,
        device=a.device,
        action_horizon=horizon,
        replan_steps=a.replan_steps,
        num_inference_steps=a.num_inference_steps,
        sigma_shift=None,
        seed=None,
        text_ckpt_dir=str(cfg.paths.text_encoder),
        text_max_len=128,
        n_cameras=n_cameras,
        num_video_frames=num_video_frames,
    )
    policy.reset()  # episode 0
    lock = threading.Lock()
    print(f"[server] READY | action_dim={policy.processor.shape_meta['action'][0].get('shape', 14)} "
          f"| code_expert={policy.model.use_code_expert}", flush=True)

    def _decode(b64):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.uint8)

    def act(body):
        instruction = body.get("instruction") or a.instruction
        imgs = body["images"]
        state = np.asarray([float(x) for x in body["state"]], dtype=np.float64)  # raw 14-D joint state
        observation = {
            "observation": {
                "head_camera":  {"rgb": _decode(imgs["head_camera"])},
                "left_camera":  {"rgb": _decode(imgs["left_camera"])},
                "right_camera": {"rgb": _decode(imgs["right_camera"])},
            },
            "joint_action": {"vector": state.tolist()},
        }
        with lock:  # single GPU model -> serialize inference
            rel = np.asarray(policy._infer_action_chunk(observation, instruction), dtype=np.float64)  # [T,14] relative-joint, denormalized
            policy.step_count += 1
        # RelativeJointTransform.backward: absolute qpos target = relative_action + current_state
        abs_qpos = rel + state[None, :]
        return {
            "action_chunk": abs_qpos.tolist(),
            "horizon": int(abs_qpos.shape[0]),
            "action_dim": int(abs_qpos.shape[1]),
            "replan_steps": int(policy.replan_steps),
            "action_type": "qpos_abs",
            "instruction": instruction,
            "code": None,  # generated policy code is printed to the server log (step 0 of each episode)
        }

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "ckpt": a.ckpt, "task": a.task,
                                 "code_expert": bool(policy.model.use_code_expert),
                                 "action_horizon": policy.action_horizon,
                                 "replan_steps": policy.replan_steps, "n_cameras": n_cameras,
                                 "action_type": "qpos_abs", "action_dim": 14})
            else:
                self._send(404, {"error": "GET /health only"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            try:
                body = json.loads(raw) if raw else {}
            except Exception as e:
                return self._send(400, {"error": f"bad json: {e}"})
            try:
                if self.path == "/reset":
                    with lock:
                        policy.reset()
                    self._send(200, {"ok": True, "reset": True, "episode": policy.episode_count})
                elif self.path == "/act":
                    self._send(200, act(body))
                else:
                    self._send(404, {"error": "POST /act or /reset"})
            except Exception as e:
                self._send(500, {"error": str(e), "trace": traceback.format_exc()[-1200:]})

    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"[server] listening on http://{a.host}:{a.port}  (POST /act, POST /reset, GET /health)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
