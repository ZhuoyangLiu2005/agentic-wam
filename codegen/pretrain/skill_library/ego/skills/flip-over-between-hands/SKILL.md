---
name: flip-over-between-hands
description: Turn an object face-over by passing it between the hands - one takes it, the other lets go, a half-turn, and the other takes the new face.
---

# flip-over-between-hands

```python
flip_over_between_hands(robot, object_name, axis, angle_deg, side, other_side)
```

## When to use

- The sub-task reads "Flip (over) the `<X>`" and the frames show BOTH
  hands involved: one lets go so the object can swing through a half-turn, and
  takes it again on the other side.
- `angle_deg` is typically `180.0`; `axis` is `'lateral'` for a flat object
  flipped toward/away, `'object_long'` for a book or lid swung about its
  spine, `'vertical'` for a turn about the upright.
- Both hands are on the object at the end -- write the release in a later
  slice.

## When NOT to use

- ONE hand does the whole turn and then presents the new face ->
  `turn_over_object` (`rotate` + `goto_relative`, contact-transparent). That
  is the "Turn over the `<X>`" subtask and it makes no claim about the other
  hand.
- The object is taken and turned but not passed -> `rotate_grasped_object`.
- The object changes hands in free space with no turn -> the `handover`
  primitive.
- The flip ENDS by seating the object (a lid flipped shut onto its box) ->
  follow the call with `attach`/`place`.
- The object is let go after the turn rather than re-taken -> write the
  `rotate` and `release` yourself.

## Hand contract (derived)

```
close(<side>) -> open(<other_side>) -> close(<other_side>)
```
Three events across two hands. `rotate` is contact-transparent in the ego API
(the grip state does not change either way), so the half-turn contributes
nothing -- the contract IS the hand swap, which is the point.

## Known limits

- A bare `rotate` followed by an open is deliberately NOT this skill: that
  would be a rename of the `rotate` primitive. The hand swap is what
  distinguishes "flip" from "turn".
- The two hands' roles are asymmetric in the body but often symmetric in the
  frames; picking the wrong `side` still produces a valid contract with the
  hands swapped.
- The final `grasp(other_side)` is one of several possible continuations;
  where the video shows the object let go after the turn, write the release
  yourself rather than calling this.
