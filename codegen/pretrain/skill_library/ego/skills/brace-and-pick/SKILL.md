---
name: brace-and-pick
description: Hold the source still with one hand and pick the object off/out of it with the other; both hands end closed.
---

# brace-and-pick

```python
brace_and_pick(robot, object_name, side, support_name, brace_side)
```

## When to use

- A subtask like "Pick up the `<X>` from the `<Y>`" where the video shows
  the second hand holding `<Y>` down — a spoon out of a container, a cap off a
  bottle body, a pill out of a bottle, a piece off a board.
- The braced counterpart of `pick_and_bring`: use it whenever the source
  would otherwise come away with the object.
- Example pairs: ('spoon', 'container'), ('bottle cap', 'bottle body'),
  ('pill', 'pill bottle'), ('game piece', 'board'), ('test tube', 'rack').

## When NOT to use

- Nothing needs bracing — the source is heavy or fixed: `pick_and_bring`, or
  the bare `pick` primitive when the hand does not travel on.
- The part comes away by a straight PULL rather than a lift: `pull_part_free`.
- The part unthreads: `unscrew_cap`.
- The brace hand is the one doing the taking: swap `side`/`brace_side`.

## Hand contract (derived)

    close(<brace_side>) -> close(<side>)

Both hands end CLOSED. The brace OUTLIVES the skill — this body emits no
release for `brace_side`; write it where the video shows that hand leaving.

Contract collision: `pull_part_free` derives the same
`close(<brace_side>) -> close(<side>)` (its third call is contact-transparent).
Choose by the motion: a LIFT clear (`pick`) here, a directed PULL there.

## Known limits

- The contract is derived from the code, not observed: a brace that never
  visibly changes hand aperture is hard to confirm from video alone.
- No release for either hand: a slice that also puts the object down is not
  this skill.
