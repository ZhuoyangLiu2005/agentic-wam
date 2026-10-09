---
name: brace-and-pull-open
description: Brace the body with one arm, take its lid/cap/flap/handle with the other, draw it open, then let the brace go - the "Open the <container>" sub-task.
---

# brace-and-pull-open

```python
brace_and_pull_open(robot, anchor_name, handle_name, brace_side, open_side, open_delta)
```

## When to use

- "Open the <drawer / lid / bag / folder / cap / packet>" sub-tasks where
  a one-armed pull would drag the whole object.
- The shape is two grasps then a pull: `grasp -> grasp -> servo_ee_delta`,
  with the brace released at the end.
- Typical pairs: pen/cap, storage box/drawer handle, appliance/lid,
  paper bag/mouth, notebook/cover.

## When NOT to use

- The closure swings shut instead of open -> `push_closure_shut`.
- One arm suffices because the body is heavy or fixed (a drawer in furniture)
  -> write `robot.grasp(handle, side); robot.servo_ee_delta(side, delta)`
  inline, or use `push_closure_shut` with an outward delta.
- A screw cap is being twisted off -> `ratchet_twist_off`.
- The part comes fully AWAY and is carried off -> `free_hand_and_pick` /
  `relocate_object` after this call.
- A zip is being run open -> `run_zipper_along_seam`.

## Gripper contract

```
close(<brace_side>) -> close(<open_side>) -> open(<brace_side>)
```

Two DISTINCT arms (the checker enforces distinctness). The `servo_ee_delta`
pull is gripper-transparent. Ends holding the opened part on `open_side`.

## Known limits

- The single-arm `grasp -> servo_ee_delta -> release` form of opening is
  identical to `adjust_in_place`; this skill deliberately covers the two-arm
  braced variant only.
- One pull only. Lids that need two pulls need two calls.
- The brace release at the end is right for caps and covers; for a drawer the
  brace often stays on. Write the three-call form inline in that case.
