---
name: scrub-back-and-forth
description: Bring the already-held cloth to a detected surface and scrub it once back and forth. Gripper-transparent; call again for more strokes.
---

# scrub-back-and-forth

```python
scrub_back_and_forth(robot, surface_name, wipe_side, stroke_delta, return_delta)
```

## When to use

- "Wipe the <surface>", "Clean ...", "Pat dry ...", "Dust ..." sub-tasks where
  the cloth/duster/sponge is ALREADY in the gripper.
- The signature is the pair of opposed `servo_ee_delta` calls after driving
  the cloth onto the detected surface.
- Typical tool/surface pairs: `rag`/`plate`, `cloth`/`screen`,
  `duster`/`monitor`, `paper towel`/`fruit`.

## When NOT to use

- The cloth still has to be picked up -> `free_hand_and_pick` first (a
  separate sub-task).
- The object is brought TO a stationary cloth instead -> the same body works,
  but `surface_name` must then name the cloth and the moving arm holds the
  object.
- Debris is being swept along rather than scrubbed in place -> not modelled.
- A held stirrer circulates inside a vessel -> `stir_braced_vessel`.

## Gripper contract

```
(gripper-transparent)
```

EMPTY. The cloth is held throughout and never let go mid-wipe, so the skill
composes into any sub-task without perturbing its gripper events.

## Known limits

- ONE stroke pair per call; call again for more.
- Takes `detections[0]` with no score gate.
- Shares its `detect -> get_robot_state -> move_arm` head with
  `lower_onto_target`; the tail (two opposed strokes vs one descent)
  distinguishes them.
- Both deltas are caller-supplied and open-loop.
