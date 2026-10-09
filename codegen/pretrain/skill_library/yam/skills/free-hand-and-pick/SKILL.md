---
name: free-hand-and-pick
description: Open the idle arm, then pick the object up with the working arm - the two-call "Pick up the <object>" sub-task.
---

# free-hand-and-pick

```python
free_hand_and_pick(robot, object_name, pick_side, free_side)
```

## When to use

- The sub-task is "Pick up the <object> ..." (or "... from the <container>")
  AND the other arm is currently holding or bracing something it must let go
  of first.
- The release-then-pick pair is the signature: write them in the SAME
  sub-task, not in two.

## When NOT to use

- Nothing is being released first -> write the bare `robot.pick(...)`; wrapping
  one primitive buys nothing.
- The arm has to be walked onto a sampled grasp pose before it can close ->
  `approach_best_grasp`.
- The object is grasped but NOT lifted away (a brace / fixture hold) ->
  `robot.grasp(...)`.
- The object is carried somewhere in the same sub-task -> `relocate_object`.
- The released arm and the picking arm are the SAME arm (drop-and-repick): this
  skill declares two DISTINCT arms and the contract checker enforces that;
  write the two calls inline instead.

## Gripper contract

```
open(<free_side>) -> close(<pick_side>)
```

Two arms, one event each. `release` contributes the open, `pick` the close.

## Known limits

- Says nothing about WHY the idle arm opens — usually a brace being let go,
  sometimes the previous item being dropped.
- A `grasp(brace) -> release(brace) -> pick(obj)` variant is not modelled;
  call `robot.grasp(...)` yourself first.
