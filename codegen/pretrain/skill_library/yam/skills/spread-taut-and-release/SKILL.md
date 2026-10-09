---
name: spread-taut-and-release
description: With both arms already gripping, drag them apart to pull the item taut, then release both - the flatten / smooth-out sub-task unit.
---

# spread-taut-and-release

```python
spread_taut_and_release(robot, left_delta, right_delta)
```

## When to use

- A "Flatten out the <garment>", "Smooth out ...", "Straighten out ..."
  sub-task where both arms already have hold of the item.
- Opposite-sign deltas are the signature: the two grips move APART so the cloth
  goes taut, then both open.
- Typical items: trousers, shirts, towels, blankets, scarves, jackets.

## When NOT to use

- The grip has to be taken first AND an edge carried across -> `fold_edge_over`
  (its body includes the two grasps).
- One arm only -> `adjust_in_place`.
- The item is being wound up rather than pulled flat -> `roll_up_one_bite`.
- Alternating one-hand presses on the fabric -> `alternating_smoothing_pats`.

## Gripper contract

```
open(left) -> open(right)
```

Concrete arms, opens only -- the closes belong to whatever established the grip
(a `fold_edge_over`, two `robot.grasp` calls, or the previous sub-task).

## Known limits

- Assumes the grip already exists. Calling it with empty grippers is a no-op
  followed by two spurious opens.
- ONE spread per call; re-grip and call again for more.
- Fixed left-then-right ordering, like `fold_edge_over`.
- Both deltas are caller-supplied and open-loop.
