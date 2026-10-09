---
name: pour-into-target
description: Bring a held vessel's mouth over a target, tip it until it empties, and bring it back upright; the grip never changes.
---

# pour-into-target

```python
pour_into_target(robot, vessel_name, target_name, side, axis, angle_deg)
```

## When to use

- A subtask like "Pour `<X>` into/onto `<Y>`" / "Pour out `<X>`" where
  the `side` hand ALREADY holds the vessel (bottle, beaker, cup, bowl, basin,
  spoon, bag).
- ONE pour per call. Write a scoop-and-pour repetition as separate subtasks.
- Example vessels: 'spoon', 'cup', 'beaker', 'test tube', 'bottle', 'bag'.
  Typical values: `axis='object_long'` (a spoon) or `'lateral'` (a bottle),
  `angle_deg=110.0`; the recovery tip is the negated angle, i.e. exactly back
  to upright.

## When NOT to use

- The hand does not yet hold the vessel: write the `pick`/`grasp` first (this
  body has no grasp line and would silently mis-describe the slice).
- The vessel is tipped and then SET DOWN inside the same slice: add your own
  `place`/`release` after the call.
- The tool is loaded from a source rather than emptied over a target (a scoop,
  a dip): `scoop_and_carry`.
- The contents are stirred or agitated rather than poured out:
  `stir_contents`.

## Hand contract (derived)

    (gripper-transparent) — no close/open events at all

`goto_relative` and `rotate` are both contact-transparent in the ego API: a
pour never changes the grip. The vessel must already be in the `side` hand
and is still there afterwards.

Contract collision: `turn_over_object` is also fully transparent.

## Known limits

- This skill declares NO contact events, so the whole body is a motion
  scaffold plus a transparent contract; nothing here can be checked against
  hand open/close events.
- Do not add stray `close_hand`/`open_hand` around a pour: the grip does not
  change during a pour.
- The tip magnitude is a semantic description of the frames, not a measured
  angle; `110.0` simply means "tip until it empties".
- Partial pours (tip, hold, tip further) are not modelled — call it again.
- The relation is hard-coded to `'above'`.
