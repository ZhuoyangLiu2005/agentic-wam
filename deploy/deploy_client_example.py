#!/usr/bin/env python
"""Minimal ROBOT-SIDE client for the AGENTICWAM-Cosmos deploy server (needs only numpy + Pillow + stdlib).

Shows the payload format + the receding-horizon control loop. Replace get_observation() /
send_qpos() with real YAM camera + robot reads. The server returns ABSOLUTE 14-D joint targets, so
the robot just commands them directly (no delta integration on the client).

Observation contract (must match training / the served arm's data config, e.g. configs/real_post_train/data/yam.yaml):
  cameras : head=cam_high (top-down), left=cam_left_wrist, right=cam_right_wrist ; HxWx3 uint8 RGB.
  state   : 14-D joint vector, layout [Ljoint0..5, Lgrip(6), Rjoint0..5, Rgrip(13)].
Action:
  action_chunk[i] : 14-D ABSOLUTE qpos target, same layout as state. Execute the first
                    `replan_steps` rows, then re-plan on a fresh observation.
"""
import base64
import io
import json
import urllib.request

import numpy as np
from PIL import Image

SERVER = "http://<MODEL_HOST>:8711"          # host:port of deploy_server.py
INSTRUCTION = "put the plastic bottles in the bin"


def b64png(rgb_uint8):
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def post(path, body):
    req = urllib.request.Request(SERVER + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def get_observation():
    # TODO: replace with real YAM reads.
    #   cameras -> HxWx3 uint8 RGB ; state -> 14-D [Ljoint0..5, Lgrip, Rjoint0..5, Rgrip].
    dummy = np.zeros((480, 640, 3), np.uint8)
    return {"head_camera": dummy, "left_camera": dummy, "right_camera": dummy, "state": [0.0] * 14}


def send_qpos(qpos_14):
    # TODO: command the bimanual YAM to the absolute joint targets (one control tick), then return.
    pass


def main():
    print("health:", urllib.request.urlopen(SERVER + "/health").read().decode())
    post("/reset", {})                                    # start a new episode/rollout
    for _control_step in range(10000):
        obs = get_observation()
        resp = post("/act", {
            "instruction": INSTRUCTION,
            "images": {"head_camera": b64png(obs["head_camera"]),
                       "left_camera": b64png(obs["left_camera"]),
                       "right_camera": b64png(obs["right_camera"])},
            "state": obs["state"],
        })
        chunk = np.array(resp["action_chunk"])            # [horizon, 14] ABSOLUTE qpos targets
        replan = int(resp["replan_steps"])
        for i in range(min(replan, len(chunk))):
            send_qpos(chunk[i])                           # 14-D absolute qpos target; wait one control tick
        # after executing `replan` rows, loop -> /act again (re-plan on fresh obs)


if __name__ == "__main__":
    main()
