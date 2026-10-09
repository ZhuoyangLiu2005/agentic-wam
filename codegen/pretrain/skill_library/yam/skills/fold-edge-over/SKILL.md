---
name: fold-edge-over
description: Both arms pinch a cloth edge, carry it across the item, and release - one fold. Repeat the call for multi-fold sub-tasks.
---

# fold-edge-over

```python
fold_edge_over(robot, item_name, left_delta, right_delta)
```

## When to use

- A "Fold the <garment/cloth>" sub-task. Folding is built almost entirely out
  of this six-call unit, repeated once per fold.
- Same-sign deltas = carry one edge across the item. Opposite-sign deltas =
  bring the two ends to the middle.
- Typical items: towels, t-shirts, napkins, trousers, blankets, scarves.

## When NOT to use

- The arms are ALREADY holding the item and the job is to pull it flat, not to
  carry an edge -> `spread_taut_and_release`.
- The cloth is being wound onto itself -> `roll_up_one_bite`.
- One arm only -> `adjust_in_place`.
- One arm pinning while the other creases -> not modelled; write it inline
  (`grasp / servo_ee_delta / open_gripper / servo_ee_delta / close_gripper`).

## Gripper contract

```
close(left) -> close(right) -> open(left) -> open(right)
```

Concrete arms, no side parameters. The two `servo_ee_delta` carries are
gripper-transparent.

## Known limits

- ONE fold per call; call again for each further fold (skill bodies cannot
  loop).
- The grasp order is fixed left-then-right. The order does not change the
  physical fold but it DOES change the derived contract, so a right-first
  realization will not match this contract.
- No regrasp mid-fold; add `robot.regrasp(item_name, side)` between calls
  when the grip slips.
