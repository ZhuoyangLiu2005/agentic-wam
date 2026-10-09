---
name: move-item-two-handed
description: Take a bulky item with both hands where it sits, carry it along a semantic direction, and let go with both.
---

# move-item-two-handed

```python
move_item_two_handed(robot, item_name, side, brace_side, direction, brace_direction)
```

## When to use

- A subtask like "Move the `<X>` (to the designated area)" / "Shift the
  `<X>`" where the video shows BOTH hands on one bulky item and NO named
  destination object — the displacement itself is the meaning.
- Both hands take it WHERE IT SITS (`grasp`, not `pick`): such items are
  usually slid or dragged rather than lifted clear.
- Pass the same value for `direction` and `brace_direction` for a straight
  carry; opposite values spread the item out.
- Example items: 'coat', 'shoe', 'shirt', 'box', 'board', 'sheet of paper'.
  Example direction: `'away_from_body'`.

## When NOT to use

- One hand does the moving: use `pick_and_bring` + `place_at_target`, or
  spell out `pick` / `move_arm` / `place`.
- The destination is a NAMED object: `pick_and_bring` (`goto_relative`
  expresses the relation; `move_arm` cannot).
- The item is folded over on itself: `fold_over_and_crease` (same opening
  calls, crease-press tail).
- The item is lifted clear of its support: use `pick`, not `grasp`.

## Hand contract (derived)

    close(<brace_side>) -> close(<side>) -> open(<side>) -> open(<brace_side>)

Both hands close, both open. Nothing is held before or after.

Contract collision: `close_closure` derives the same close/close/open/open
shape (its middle calls are `rotate` rather than `move_arm`). Choose by the
motion: a translation here, a swing there.

## Known limits

- A flat-handed push on a coat or a board changes hand aperture hardly at
  all, so the four contract events are derived from the code only.
- **`move_arm` has no destination.** If the annotation names where the item
  goes, this skill loses that information — prefer `pick_and_bring` +
  `place_at_target` when a target object is visible.
- `distance` is deliberately left unset: the ego API has no coordinates.
