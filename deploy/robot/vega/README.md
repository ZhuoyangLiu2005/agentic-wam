# Vendored T-Rex robot code (vega + Sharpa)

Copied from T-Rex `hardware_code/` at commit `2a891d3ae48d7f78e9b42035987204bf97de2db9` (MIT, `LICENSE`), so the
Sharpa deploy client runs without a T-Rex checkout. Used by `deploy/robot/vega_robot.py`.

| File | Source | Changes |
|---|---|---|
| `arm_hand_control.py` | `teleop/arm_hand_control.py` | `teleop.` imports made relative |
| `robot_descriptions.py` | `teleop/robot_descriptions.py` | `project_root` = this directory (was `parents[1]`) |
| `ik_utils.py` | `teleop/ik_utils.py` | `teleop.` import made relative (only needed because `arm_hand_control` imports it) |
| `head_camera_receiver.py`, `wrist_camera_receiver.py` | `camera/` | none |
| `config/default.yaml` | `config/default.yaml` | **meant to be edited** (site config read by `vega_robot.VegaRobotConfig.from_yaml`); environment set to the paper-eval bench (table 0.76, walls 0.60 / 0.75) |
| `third_party/sharpa-urdf-usd-xml/` | same path | only `wave_01/*/…_with_wrist.xml` and the meshes it references (collision model); Apache-2.0, `LICENSE.txt`/`NOTICE.txt` |

Do not edit these files (except `config/default.yaml`); re-copy them from T-Rex instead. Check that they still match with:

```bash
for f in arm_hand_control robot_descriptions ik_utils; do diff <T-Rex>/hardware_code/teleop/$f.py deploy/robot/vega/$f.py; done
```

Each diff should show only the lines in the table.

The packages are installed, not vendored. They come from T-Rex `hardware_code/requirements.txt`:
- `pip install -e <T-Rex>/hardware_code/third_party/dexcontrol -e <T-Rex>/hardware_code/third_party/dexmate-urdf`
- the Sharpa Wave SDK (`sharpa`)
- `pin`, `coal`, `pin-pink`, `qpsolvers`, `daqp`, `ompl`, `ruckig`, `loop-rate-limiters`, `pyzmq`, `opencv-python`, `loguru`
