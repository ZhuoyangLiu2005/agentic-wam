---
name: push-closure-shut
description: Grip a drawer, lid, door or cover and drive it along its travel until shut, then let go - the "Close the <thing>" sub-task.
---

# push-closure-shut

```python
push_closure_shut(robot, closure_name, side, close_delta)
```

## When to use

- "Close the <drawer / lid / folder / door / case>" sub-tasks.
- The object has ONE degree of freedom and the delta runs along it. Deltas are
  larger than an adjust's because the travel is the closure's, not a nudge.
- Typical `closure_name` values: `lid`, `drawer`, `drawer handle`, `folder`,
  `box cover`, `case`.

## When NOT to use

- The thing is being pulled OPEN -> `brace_and_pull_open`.
- A free object is being nudged on the surface -> `adjust_in_place` (identical
  call shape; see the collision note).
- A cap is THREADED on -> `ratchet_turn_fastener`.
- A cover has to be picked up before it can be pushed on -> write
  `pick -> servo_ee_delta` inline, or `free_hand_and_pick` then this.
- The cover has to be pressed home to latch -> `robot.attach(...)`.

## Gripper contract

```
close(<side>) -> open(<side>)
```

**Contract collision (declared):** identical to `adjust_in_place` and
`relocate_object`, and the BODY is identical to `adjust_in_place`. They are
kept separate because the sub-task differs (drive a one-DOF closure through
its travel vs reposition a free object) and the name carries the semantics.

## Known limits

- ONE push. A stiff drawer may need another; call again.
- No brace. Light boxes slide away; add `robot.grasp(body, other_side)` first.
- The release at the end is right for drawers and covers; a latch that must be
  held while it snaps needs `robot.attach(...)` instead.
- `close_delta` is caller-supplied and open-loop.
