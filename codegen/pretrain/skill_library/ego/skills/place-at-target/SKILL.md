---
name: place-at-target
description: Set the carried object down at a target with the working hand and free the second hand as well; both hands end open.
---

# place-at-target

```python
place_at_target(robot, object_name, target_name, side, free_side)
```

## When to use

- A subtask like "Place the `<X>` (in|on|at) the `<Y>`" / "Put the `<X>`
  back" where BOTH hands come off at the end — the second hand was steadying
  the object, the target, or a co-carried load.
- Pass `free_side == side` for a one-handed place; the trailing `release` is
  then a harmless re-open of an already-open hand.
- Use `'table'` for `target_name` when the video shows no finer target.
  Example pairs: ('game piece', 'board'), ('paper cup', 'stack of cups'),
  ('pill', 'pill box'), ('egg', 'egg carton'), ('phone', 'table').

## When NOT to use

- The object goes INTO a receptacle and the hand withdraws:
  `insert_into_receptacle`.
- Seating needs contact force and the object ends FIXED to the target: the
  `attach` primitive (or `press_onto_target` when the seat is a press).
- The object is only carried over the target and NOT let go:
  `pick_and_bring`.
- Nothing is held — the hand just opens: the bare `release` primitive.
- Two objects set down, or a set-down followed by moving on
  (`place -> goto_relative`): spell those out.

## Hand contract (derived)

    open(<side>) -> open(<free_side>)

`place` opens the working hand; `release` opens the second one. Nothing is
held afterwards. The skill assumes the object is ALREADY in the `side` hand.

## Known limits

- The two `open` events are derived from the code; a release that opens the
  hand only slightly is hard to confirm visually.
- `target_name = 'table'` is a fallback, not a perceived target.
