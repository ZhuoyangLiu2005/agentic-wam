---
name: insert-into-receptacle
description: Seat the held object into a receptacle, let go, and withdraw the now-empty hand along a semantic direction.
---

# insert-into-receptacle

```python
insert_into_receptacle(robot, object_name, target_name, side, direction)
```

## When to use

- A subtask like "Insert the `<X>` into the `<Y>`" / "Put the `<X>` in
  the `<Y>`" where the `side` hand ALREADY holds the object and LETS GO once
  it is seated, then withdraws.
- Also fits "Place the `<X>` in the `<Y>`" labels when the motion is really a
  seating into a receptacle.
- Example pairs: ('battery', 'remote control'), ('cable plug', 'router'),
  ('book', 'book cover'), ('photo', 'envelope'), ('sock', 'shoe'). Example
  `direction='toward_body'`.

## When NOT to use

- **The hand keeps holding after seating.** The ego API is explicit here:
  write `robot.goto_relative(side, target_name, 'inside')` instead.
- The object ends FIXED to the target rather than resting in it: the `attach`
  primitive.
- Seating needs a force push: `press_onto_target`.
- The object is set down at/onto a target rather than into it:
  `place_at_target`.
- The receptacle must be braced: `align_and_seat_braced`, or add your own
  `grasp(target_name, brace_side)` before the call.
- The slice is just the single `insert` with no withdrawal: call the
  primitive directly.

## Hand contract (derived)

    open(<side>)

`insert` opens (it asserts the letting-go); `move_arm` is
contact-transparent. The hand is EMPTY afterwards.

## Known limits

- The single `open` is derived from the code; a release inside a tight
  receptacle barely changes the visible hand aperture.
- `direction` is the withdrawal of the hand, not of the object;
  `'toward_body'` is the egocentric "back toward me".
