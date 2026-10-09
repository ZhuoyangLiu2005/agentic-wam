---
name: hand-over-hand-extract
description: Two grip-pull-release bites that walk a long object out of a bag, sleeve or slot — the "take out the X from the Y" sub-task.
---

# hand-over-hand-extract

```python
hand_over_hand_extract(robot, object_name, side, pull_delta)
```

## When to use

- A sub-task like "Take out the <object> from the <container>": a racket from
  its bag, a glove from a storage bag, a note off a board, bread from a paper
  bag, a sleeve off a cup.
- Whenever ONE pull cannot clear the object and the arm lets go, re-bites
  further along, and pulls again.

## When NOT to use

- One pull is enough: `pull_part_free` (the "remove the <part>" family).
- The object lifts straight out and is then carried: `pick`.
- Unplugging something that needs a button held first (`press -> pick`) —
  issue the `press` in the calling program.
- Extracting by unscrewing: `ratchet_twist_off`.

## Gripper contract

```
close(<side>) -> open(<side>) -> close(<side>) -> open(<side>)
```
Ends OPEN: the object is out but not held.

## Known limits

- Two bites per call (skill bodies cannot loop); a deep bag needs repeated
  calls.
- Both bites re-`grasp` the SAME `object_name`, so the detector must still
  find it once it is partly out; switching to a sub-part name (e.g. a handle)
  between bites needs two calls.
- The container is normally braced by the other arm; that brace is a separate
  sub-task and stays out of this contract.
- Contract `close -> open -> close -> open` collides with `ratchet_twist_off`;
  the discriminator is `servo_ee_delta` vs `rotate`.
