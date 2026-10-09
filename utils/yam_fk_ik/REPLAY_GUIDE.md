# YAM 64-D EEF sample — real-robot IK replay & validation guide

One ABC-bottles episode exported in three synchronized representations. Kinematics use the **real
station URDF** (`station/station/station.xml`) — FK matches the recorded EEF **exactly (0.0 mm /
0.0°, absolute world frame)**, with **no joint sign flip** and EEF = `*_grasp_site`.

## Files (`sample_trajectory/`)
| file | shape | meaning |
|---|---|---|
| `cam_high.mp4`, `cam_left_wrist.mp4`, `cam_right_wrist.mp4` | video | 3 camera views (30 fps) |
| `raw_state_14d.npy` | [T,14] | recorded ABSOLUTE joints (state) |
| `raw_action_14d.npy` | [T,14] | recorded ABSOLUTE joint TARGETS (action) |
| `state_64d.npy` | [T,64] | unified 64-D state (absolute EEF, world frame), **un-normalized** |
| `action_delta_64d.npy` | [T,64] | unified 64-D action = 1-step delta rel. current state, **un-normalized** |
| `ik_replay_joints_14d.npy` | [T,14] | joints recovered from `action_delta_64d` via IK (round-trip check) |
| `meta.json` | — | model, layout, IK round-trip error |

## Layout
- **14-D joints**: `[Ljoint0-5, Lgrip(idx6), Rjoint0-5, Rgrip(idx13)]`.
- **64-D per arm**: `[ eef 9-D (pos3 + rot6d) | hand 22-D = 0 | gripper 1-D ]`; L = `0:32`, R = `32:64`.
  `state` = absolute EEF (world); `action` = delta EEF in the current-EEF frame.

## Forward (how the 64-D was made)
```
joints --> station FK (station.xml, *_grasp_site; qpos left[0:6] right[8:14]) --> abs EEF 4x4 (world)
       --> [pos3, rot6d(first two cols)] = 9-D           (YamUnifiedEEFTransform)
action delta = (T_current_EEF)^-1 · T_action_EEF
```
No sign flip — the station model uses the same joint convention as the recorded data.

## Replay on the real robot (backward = IK)
```python
import numpy as np, yam_kinematics as K          # ../yam_kinematics.py (needs mujoco)
a = np.load("action_delta_64d.npy"); s = np.load("raw_state_14d.npy")
q = s[0].copy()
for t in range(len(a)):
    for arm, js, je, gi, off in [("left", 0, 6, 6, 0), ("right", 7, 13, 13, 32)]:
        T_cur    = K.fk(q[js:je], arm)
        T_target = K.apply_delta_9d(T_cur, a[t, off:off+9])   # 9-D delta -> abs EEF target
        ok, qj   = K.ik(T_target, arm, init6=q[js:je])        # -> joints
        q[js:je] = qj
        q[gi]    = a[t, off+31]                                # absolute gripper
    send_to_robot(q)
```

## Validate before the robot
1. `python ../yam_kinematics.py` → FK/IK self-test (round-trip ~0).
2. `ik_replay_joints_14d.npy` vs `raw_action_14d.npy` should match to ~milli-radian
   (`meta.json → ik_roundtrip_*`).

## Notes
- Absolute EEF (`state_64d`) is now in the true world frame — matches the recording exactly (station has the real
  base mounts + link geometry). The earlier ~cm link-geometry offset is gone.
- Joint convention is assumed the same as the recorded robot; this replay confirms it on hardware.
