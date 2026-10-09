---
name: pick-and-bring
description: Pick an object up with one hand and carry it to a semantic relation on where it is going next; the hand ends holding it.
---

# pick-and-bring

```python
pick_and_bring(robot, object_name, side, target_name, relation)
```

## When to use

- A subtask like "Pick up the `<X>`" / "Take the `<X>`" where the video
  shows the lift AND the hand travelling on toward where the object is going.
  The whole thing is ONE annotated slice, which is exactly what this body is
  sized for.
- Example objects: 'game piece', 'paper cup', 'pill box', 'test tube', 'egg',
  'lid', 'bottle'. Typical relation: `'above'`; `'in_front_of'` and
  `'inside'` also fit.

## When NOT to use

- The lift needs the other hand to hold the source still: `brace_and_pick`.
- The hand closes on something it will NOT carry away: that is the `grasp`
  primitive (or `rotate_grasped_object` if it then turns).
- The object is set down inside the same slice: `place_at_target`,
  `insert_into_receptacle` or `attach`.
- Nothing is lifted and the hand only travels: bare `goto_relative`.
- Two objects are picked up in one slice: spell out the two `pick` calls.

## Hand contract (derived)

    close(<side>)

`pick` closes; `goto_relative` is contact-transparent. The hand is STILL
HOLDING the object when the skill returns — the caller owns the put-down.

Contract collision: `rotate_grasped_object` derives the same single
`close(<side>)`. They are distinguished by their bodies, not their contracts.

## Known limits

- The contract is a code-level claim; a real pinch pick may show little
  visible change in hand aperture.
- `relation` is a semantic description of the frames, never a measured
  offset; `clearance` is left unset on purpose.
