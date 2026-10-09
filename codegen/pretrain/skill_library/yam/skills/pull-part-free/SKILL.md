---
name: pull-part-free
description: Grip a part where it sits, drag it straight off along one bounded delta, release — the whole of a "remove the <part>" sub-task.
---

# pull-part-free

```python
pull_part_free(robot, part_name, side, pull_delta)
```

## When to use

- A sub-task like "Remove the <part> from the <thing>": a wrapper off a
  straw, a battery cover off a device, an insole out of a shoe, a part off a
  fixture.
- Anything that comes off by a single directional pull rather than a lift — the
  separation axis is expressed as `pull_delta`, which is why this is not
  `pick`.

## When NOT to use

- The part needs SEVERAL bites to clear a container: `hand_over_hand_extract`
  (the "take out ... from ..." family).
- It unscrews: `ratchet_twist_off`.
- You want to carry the freed part somewhere: that transport is `place` /
  `handover` in the calling program; this sub-task ends with the part let go
  where it came free.
- Lifting an object straight up off a surface with no separation axis: `pick`.

## Gripper contract

```
close(<side>) -> open(<side>)
```
`servo_ee_delta` between them is gripper-transparent.

## Known limits

- A bare `grasp -> release` with no pull is deliberately NOT what this skill
  encodes.
- Contract `close -> open` is shared with `adjust_in_place`; the discriminator
  is the sub-task and the pull direction.
- No force sensing: a still-attached part is pulled regardless.
