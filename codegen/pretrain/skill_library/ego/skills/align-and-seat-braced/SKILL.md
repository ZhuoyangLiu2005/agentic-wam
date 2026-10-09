---
name: align-and-seat-braced
description: The two-hand "align and insert/fit <part> into <target>" subtask - one hand pins the target so it cannot shift, the other pushes the part home and lets go.
---

# align-and-seat-braced

```python
align_and_seat_braced(robot, object_name, target_name, side, brace_side)
```

## When to use

- The sub-task reads "Align and insert/fit `<part>` into `<target>`" (or
  "Align, insert and press ...") and the frames show ONE hand steadying the
  target while the other registers the part and pushes it home in the same
  slice.
- The alignment and the seating are not separable in the slice: write them as
  this one call, not as an approach followed by a later seating.

## When NOT to use

- Nothing needs holding (the target is heavy, clamped, or the receptacle is
  fixed) -> `insert_into_receptacle`, or the bare `insert` primitive.
- The part ends FIXED by contact force onto a surface rather than seated in a
  socket -> write `robot.grasp(target, brace_side)` + `robot.attach(part,
  target, side)` inline; swapping the seating verb would misstate the effect.
- The part must be turned to match the target first -> add a
  `robot.rotate(part, axis, angle, side)` line before the call.
- "Align/organize a STACK" (paper, documents) -- write that as `move_arm`
  nudges on both hands, not as a seating.
- The hand keeps holding after seating -> `goto_relative(side, target,
  'inside')`; `insert` asserts the release.

## Hand contract (derived)

```
close(<brace_side>) -> open(<side>)
```
The brace hand closes and STAYS closed (no release in this body, on purpose).
The `side` hand enters holding the part and leaves empty -- that open is
`insert`'s declared effect.

## Known limits

- The seating verb is baked in as `insert`. A slice whose seating is `attach`
  or `press` must be written inline -- those are different declared effects,
  not stylistic variants.
- No rotation step is included; add the `rotate` line yourself when the video
  shows the part turned to match.
- The brace hand's release is deliberately absent; a program that never
  releases it will under-report an open event.
