---
name: pull-open-and-hold
description: Take hold of a handle, leaf or free end where it sits and draw it open along a semantic direction, keeping hold of it - for things that stay attached.
---

# pull-open-and-hold

```python
pull_open_and_hold(robot, object_name, direction, side)
```

## When to use

- The sub-task is "Pull open the `<X>`" / "Pull the `<X>` from the `<Y>`" /
  "Pull tight the `<X>`" and the frames show one hand taking a handle, leaf,
  leg or free end and drawing it along a direction while it stays attached to
  whatever it came from (a drawer, a stand, a door, tape off a roll, a ribbon
  pulled taut).
- The hand is still on it at the end of the sub-task.
- Two hands drawing OPPOSITE ways (a frame pulled open, a drawstring pulled
  taut, a napkin stretched) = TWO calls, one per hand with opposite
  `direction`s.

## When NOT to use

- The part comes AWAY from an assembly -> `pull-part-free`. That is the
  "Remove the `<X>` from the `<Y>`" sub-task and it brings its own brace hand;
  this skill has none because a drawer or a door does not need one.
- The item is carried to a new place -> `move-item-two-handed` or
  `pick-and-bring`.
- The thing swings on a hinge and is best described as an angle ->
  `open-closure` (e.g. a door written as `grasp` + `rotate(...,'vertical',90.0,...)`).
- A cap is being turned off a thread -> `unscrew-cap`.

## Hand contract (derived)

```
close(<side>)
```
One event. `move_arm` is contact-transparent, and the hand deliberately ends
CLOSED -- write the release, the re-bite or the second pull the video shows as
your own next line.

## Known limits

- It is a strict sub-sequence of `pull-part-free`. The boundary is intent +
  arity, not structure: `pull-part-free` is a BRACED two-hand removal that
  ends with the part in hand; this is a one-hand draw on something that stays
  attached.
- No trailing release or re-grip is included; what follows a pull varies and
  must be written by the caller.
- `move_arm` distance is unset; how far the thing opened is unmodelled.
