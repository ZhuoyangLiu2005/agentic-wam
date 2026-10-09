---
name: press-onto-target
description: Press a target with both hands and come off it — seating a lid/case/tag home by force, or working a control.
---

# press-onto-target

```python
press_onto_target(robot, target_name, side, other_side)
```

## When to use

- A sub-task like "Press the `<X>` (onto the `<Y>`)" / "Push the button"
  where the hands contact the target and come straight off it.
- Both hands press in turn (`press -> press`), e.g. seating a lid by pushing
  down on each side.
- Pass `other_side == side` for a ONE-handed press: the second call is then a
  second dab, which is how a repeated button push is written.
- Example targets: 'lid', 'snap-on cover', 'button', 'keyboard', 'capsule'.

## When NOT to use

- The object is HELD and pressed home while still gripped: use `attach` (place
  whose seating needs force, ending fixed) — `press` holds nothing before or
  after.
- The press is a crease at the end of a fold: `fold_over_and_crease` already
  contains it.
- The closure swings shut: `close_closure`.
- More than a handful of repeats (a keyboard, a pump): write a literal
  `for _ in range(n): robot.press(...)` loop in the program — this body is
  fixed at two dabs.

## Hand contract (derived)

    close(<side>) -> open(<side>) -> close(<other_side>) -> open(<other_side>)

`press` is a close-then-open contact pair per the ego contract. Both hands are
clear before and after; nothing is held.

## Known limits

- A press with a flat palm or one fingertip barely changes hand aperture, so
  the four contract events are hard to confirm from video alone.
- Many presses are a single primitive call and need no skill.
- Fixed at two dabs; repeated presses belong in a program-level loop.
- The body cannot express a press whose target is HELD by the other hand —
  add your own `grasp` before it (`grasp -> press`).
