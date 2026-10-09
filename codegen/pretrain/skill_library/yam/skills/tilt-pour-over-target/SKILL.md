---
name: tilt-pour-over-target
description: Detect the receptacle, carry a held vessel over it, clear the rim, tip once and return to upright.
---

# tilt-pour-over-target

```python
tilt_pour_over_target(robot, vessel_name, target_name, side, aim_delta, approach_rpy, tilt_deg=90.0)
```

## When to use

- A sub-task like "Pour the <stuff> into the <receptacle>": candies, cereal,
  soil, liquid, seeds — out of a scoop, spatula, bag or packet into a
  container, pot, cup or bowl.
- The vessel is ALREADY HELD from a previous sub-task and must stay held
  afterwards.

## When NOT to use

- Tipping a vessel that is being held by BOTH arms (a big bag) — no two-arm
  variant is encoded here.
- Scooping/portioning before the pour, or setting the vessel back down after —
  separate sub-tasks (`place`, or a `servo_ee_delta` retreat).
- Emptying by lifting one edge of a box on the table rather than tipping a held
  vessel: `grasp` + `servo_ee_delta` up.

## Gripper contract

```
(gripper-transparent)
```

No gripper event at all — perception, one arm move, one servo, two rotates.
Gripper-transparent bodies are normal at this altitude and ride along
checkably-neutral for the grounder.

## Known limits

- Passing a zero `aim_delta` skips rim clearance and drives the vessel into
  the receptacle; always lift clear before tipping.
- `detect_objects_oneshot` returns ALL matches; the body takes
  `detections[0]` (highest score) without a score gate.
- One tip per call; several portions need several calls.
- `approach_rpy` is a caller-supplied absolute orientation — gripper-down
  `[180.0, 0.0, 0.0]` is typical, but a vessel gripped at an angle needs the
  current `ee_rpy` instead.
