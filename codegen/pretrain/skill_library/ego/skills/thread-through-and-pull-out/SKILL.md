---
name: thread-through-and-pull-out
description: Push a lace, cord or pin through a hole with one hand, pin the target with it, then catch the emerging end with the other hand and draw the slack through.
---

# thread-through-and-pull-out

```python
thread_through_and_pull_out(robot, object_name, target_name, pull_direction, side, other_side)
```

## When to use

- The sub-task is "Pass/thread the `<cord>` through the `<X>`" and the frames
  show the leading end going in one side of a hole and being taken out the
  other by the OTHER hand.
- The maneuver is a hand SWAP around a hole: near hand lets go, far hand
  catches. That is what makes it more than an `insert`.
- Repeat as further sub-tasks (one eyelet each), e.g. when lacing a shoe.

## When NOT to use

- The object is seated in a receptacle and stays there -> `insert` /
  `insert-into-receptacle`. Nothing is caught on the far side.
- The object is handed from one hand to the other in free space ("Pass the
  bottle", "Pass the knife") -> the `handover` primitive, optionally with a
  `move_arm` reach.
- The object is passed to another PERSON -> `move_arm(side,
  'away_from_body')`.
- Nothing comes out the far side to catch -> `insert` alone.

## Hand contract (derived)

```
open(<side>) -> close(<side>) -> open(<other_side>) -> close(<other_side>)
```
Four events across two hands, which is why this reads as a maneuver rather
than a primitive: the `insert` releases the near end, the near hand re-closes
on the TARGET, the far hand drops what it held and closes on the cord.
`move_arm` is contact-transparent.

## Known limits

- A pinch on a thin lace is barely distinguishable from an air-pinch in
  video, so the four events are hard to confirm visually.
- `pull_direction` is ego-frame; pick it from the frames.
- The skill assumes the far hand was holding something to release. Where it
  was already empty, the leading `release` over-states an open event.
