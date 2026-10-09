---
name: turn-over-object
description: Flip a held object face-over and bring the new face to a semantic relation on where it is going; the grip never changes.
---

# turn-over-object

```python
turn_over_object(robot, object_name, side, axis, angle_deg, target_name, relation)
```

## When to use

- A sub-task like "Turn over the `<X>`" / "Turn the `<X>` over" where the
  `side` hand ALREADY holds the object, flips it, and carries the new face on
  toward where it is going. The turn-over is always "so that the other face is
  now facing the work" — the presentation is part of the sub-task.
- Example objects: 'box', 'shoe', 'bottle cap', 'card', 'sock'. Typical
  values: `axis='lateral'`, `angle_deg=180.0`, `relation='above'`.

## When NOT to use

- The hand does not yet hold the object: write the `pick`/`grasp` first, or
  use `rotate_grasped_object` (which contains the grasp).
- The flip is a partial turn with no presentation: `rotate_grasped_object`.
- The object is flipped by passing it hand to hand: write that as
  `handover(obj, 'left', 'right'); handover(obj, 'right', 'left')` — it is a
  different maneuver.
- The flipped object is SET DOWN in the same sub-task: add your own `place`
  (`rotate -> place`).
- A book page being turned: that is a `grasp`/`rotate`/`release` loop.

## Hand contract (derived)

    (gripper-transparent) — no close/open events at all

`rotate` and `goto_relative` are both contact-transparent. The object must
ALREADY be in the `side` hand and is still there afterwards.

Contract collision: `pour_into_target` is also fully transparent.

## Known limits

- `angle_deg=180.0` and `relation='above'` are semantic descriptions of the
  motion, never measured.
- The object must already be held — this body has no grasp line and would
  silently mis-describe a sub-task that starts empty-handed.
