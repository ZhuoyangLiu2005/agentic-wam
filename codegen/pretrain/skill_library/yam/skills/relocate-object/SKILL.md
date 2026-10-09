---
name: relocate-object
description: One-arm transport in a single slice - pick the object up and place it at the named target. The "Move the <object> to the <target>" sub-task.
---

# relocate-object

```python
relocate_object(robot, object_name, target_name, side)
```

## When to use

- The sub-task is "Move the <object> to the <target>" / "Bring ..." /
  "Put ... on ..." and the object is LIFTED on the way.
- One arm does the whole thing inside one sub-task - that is what makes it a
  single skill rather than a separate pick and place.
- Typical `object_name` values: `container`, `box`, `book`, `folder`, `vase`,
  `bottle`; `target_name` is often the work surface itself.

## When NOT to use

- The object SLIDES without being lifted -> `adjust_in_place`
  (`grasp -> servo_ee_delta -> release`).
- Only the placement half is in your sub-task (the object is already held) ->
  `lower_onto_target`, or `robot.place(...)`.
- Only the pick half is in your sub-task -> `free_hand_and_pick`.
- The object is handed between arms -> `robot.handover(...)`.

## Gripper contract

```
close(<side>) -> open(<side>)
```

`pick` contributes the close, `place` the open.

**Contract collision (declared):** identical to `adjust_in_place` and
`push_closure_shut`. Distinguish by body and sub-task, not by contract.

## Known limits

- No brace. To steady a companion object, write
  `robot.grasp(other_object, other_side)` before this call.
- Two primitives, so the value is vocabulary (naming the sub-task) rather than
  compression.
- Says nothing about height or approach; `robot.place` owns those.
