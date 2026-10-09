---
name: fold-over-and-crease
description: Brace a flat item, carry its free edge across in two steps, let it down, and press the crease — one fold.
---

# fold-over-and-crease

```python
fold_over_and_crease(robot, item_name, side, brace_side, direction, fold_direction)
```

## When to use

- A subtask like "Fold the `<X>`" / "Fold over the `<edge>` onto `<Y>`"
  for cloth, paper, a bag or a sleeve.
- The fold ends with the crease being PRESSED flat by the working hand.
- ONE fold per call — write a multi-fold garment as one subtask per fold.
- Example items: 'paper', 'shirt', 'napkin', 'wrapping paper', 'sleeve',
  'paper bag'. Example motion: `direction='toward_body'` then
  `fold_direction='left'` (a diagonal fold).
- The brace OUTLIVES the skill; write its release where that hand leaves.

## When NOT to use

- The item is only smoothed or flattened, not folded over:
  `flatten_by_pulling_taut`.
- The fold is really a relocation of a bulky item: `move_item_two_handed`.
- The item is rolled or wrapped around something: `wrap_around_target`.
- No crease is pressed and BOTH hands leave: `move_item_two_handed` (same
  opening calls, `release -> release` tail).

## Hand contract (derived)

    close(<brace_side>) -> close(<side>) -> open(<side>) -> close(<side>) -> open(<side>)

`grasp` closes, `release` opens, `press` is a close-then-open pair (the last
two events). The `brace_side` hand closes and STAYS closed.

## Known limits

- Fabric and paper folds are a hard case for visual verification: the hand
  pinches a thin edge, so hand aperture barely moves and the contract is
  derived from the code only.
- The two `move_arm` steps describe a curved carry semantically; they are a
  scaffold, not a trajectory.
- Multi-fold garment sequences are NOT one call — call this once per
  annotated fold.
