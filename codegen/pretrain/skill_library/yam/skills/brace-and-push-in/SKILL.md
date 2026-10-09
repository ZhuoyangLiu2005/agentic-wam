---
name: brace-and-push-in
description: Grip the part twice on the same object - low to keep it aligned, high to push - and drive it home into its receptacle. The "Insert the <part>" sub-task.
---

# brace-and-push-in

```python
brace_and_push_in(robot, object_name, brace_side, push_side, push_delta)
```

## When to use

- "Insert the <part> into the <receptacle>" sub-tasks for LONG THIN parts
  that buckle under a one-armed push: a straw into a cup lid, a plug into a
  socket, a document into a sleeve, a refill into a binder.
- Both grasps name the SAME object - that is the tell that separates this from
  `brace_and_pull_open` (which braces one object and pulls a different one).

## When NOT to use

- The whole semantic insert (grasp, lift, align, descend, release, retreat) is
  the sub-task -> `robot.insert(object, target, side)`.
- The part is carried over a target and lowered in -> `lower_onto_target`.
- The part has to be fed THROUGH and out the other side (a lace, a belt, a
  cable) -> write the feed inline with primitives.
- The part threads in -> `ratchet_turn_fastener`.
- The part is seated by pressing rather than sliding -> `robot.attach(...)` or
  `robot.press(...)`.

## Gripper contract

```
close(<brace_side>) -> close(<push_side>)
```

Two DISTINCT arms (the checker enforces distinctness), two closes, no open --
this sub-task ENDS holding. The push is gripper-transparent.

## Known limits

- ONE push. A deep insertion needs a re-grip shuffle, which is NOT modelled
  here.
- Ends holding with both arms; the caller must free an arm afterwards.
