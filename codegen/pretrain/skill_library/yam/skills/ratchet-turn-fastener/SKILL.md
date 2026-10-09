---
name: ratchet-turn-fastener
description: Two ratchet strokes on a threaded fastener - grip, turn, let go, swing the wrist back, grip, turn. The open in the middle is the wrist unwind, not a drop.
---

# ratchet-turn-fastener

```python
ratchet_turn_fastener(robot, fastener_name, turn_side, angle_deg, backswing_deg)
```

## When to use

- "Screw the <lid/bulb/knob/cap> onto/into the <body>" sub-tasks, and the
  unscrewing direction too (pass a negative `angle_deg`).
- The tell is a `rotate` sandwiched between gripper events with the wrist
  returning while OPEN (`close -> rotate -> open -> rotate -> close -> rotate`).
- Typical fasteners: `knob`, `nut`, `lid`, `bulb`, `jar lid`, `bottle cap`.

## When NOT to use

- The arm still has to be driven onto the fastener -> prefix
  `approach_best_grasp` (`sample_grasp_pose / select_best_grasp / move_arm`),
  then the ratchet.
- A single in-hand twist with no re-grip -> `robot.rotate(...)` inline.
- The item is being rolled along the surface, not turned about the gripper axis
  -> `roll_up_one_bite`.
- A cap is pushed straight on rather than threaded -> `push_closure_shut` or
  `brace_and_push_in`.

## Gripper contract

```
close(<turn_side>) -> open(<turn_side>) -> close(<turn_side>)
```

Single-arm ratchet. `robot.rotate` is gripper-transparent, so only the three
grip events land in the contract. Ends CLOSED, mid-turn, on purpose.

## Known limits

- TWO strokes. A tight lid needs more; call again.
- No brace. Add `robot.grasp(body_name, other_side)` when the body being
  screwed into can spin.
- `robot.rotate` keeps the end-effector position fixed, so a fastener that must
  also descend as it threads needs a `servo_ee_delta` between calls.
- Angles are caller-supplied and open-loop.
