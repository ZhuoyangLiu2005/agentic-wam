---
name: unscrew-cap
description: Take a threaded part, twist, let the wrist unwind and re-close, twist again, and lift the freed part clear.
---

# unscrew-cap

```python
unscrew_cap(robot, cap_name, side, axis, angle_deg)
```

## When to use

- A sub-task like "Unscrew the `<X>` from the `<Y>`" for a threaded part: a
  cap, lid, nipple, pump head, bulb, or a jar/container that screws onto a
  body.
- The wrist reset between twists is written as `regrasp` — the ego API's own
  name for the momentary open-and-re-close (equivalent to a
  `close_hand`/`rotate`/`open_hand` cycle).
- Typical values: `axis='object_symmetry'` or `'vertical'`, `angle_deg=90.0`.

## When NOT to use

- **More or fewer than two twist cycles.** Skill bodies are STRAIGHT-LINE, so
  the cycle count is baked in at two. Write tighter threads as a literal
  `for _ in range(n): close_hand; rotate; open_hand` in the program when the
  video shows 3+ cycles.
- The part comes away by a straight PULL, not a twist: `pull_part_free`.
- The closure swings on a hinge: `open_closure`.
- Screwing something ON: negate `angle_deg` and drop the final `pick` — spell
  it out; it is not this skill.
- The hand only turns an already-held part: `rotate_grasped_object`.

## Hand contract (derived)

    close(<side>) -> open(<side>) -> close(<side>) -> close(<side>)

`grasp` closes, `regrasp` opens-then-closes, `pick` closes. The trailing
double close is real: `regrasp` re-closes and `pick` closes again on the freed
part. The hand ENDS HOLDING the part.

## Known limits

- A twist reset on a small cap is a fine-motor event that is hard to confirm
  from video alone.
- **The cycle count is fixed at 2** (a straight-line-body constraint). A
  2-cycle body applied to a many-cycle video under-describes the sub-task.
- **No brace.** Many real unscrews hold the body with the other hand; add your
  own `grasp(body_name, brace_side)` before the call when the video shows it.
- The final `pick` asserts the part comes AWAY. When the part is only loosened
  and left in place, drop it (or use `move_arm`).
- `angle_deg` is a semantic twist amount, not a measured one.
