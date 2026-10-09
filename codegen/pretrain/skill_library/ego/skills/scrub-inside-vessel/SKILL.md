---
name: scrub-inside-vessel
description: Clean the inside of a vessel with a brush - one hand holds the vessel still, the other puts the brush down inside and works it with a twist and a stroke pair.
---

# scrub-inside-vessel

```python
scrub_inside_vessel(robot, vessel_name, tool_name, angle_deg, stroke_direction, return_direction, side, brace_side)
```

## When to use

- The sub-task is "Brush the `<vessel>`" / "Brush up and down the
  `<vessel>`" and the frames show a brush entering a bottle, tube, jar or cup
  and being worked against the INSIDE wall.
- The off hand is holding the vessel (a free-standing vessel would otherwise
  travel under the brush).
- Wrap in a program-level `for _ in range(n):` for a full clean.

## When NOT to use

- The stroke is over an OPEN surface (plate, tray, shoe, a sauce being
  brushed onto food) -> `wipe-surface-back-and-forth`. It needs no brace and
  no twist.
- Nothing holds the vessel (it is clamped, or sitting in a rack) -> drop the
  brace and call `wipe-surface-back-and-forth` with `relation='inside'`.
- The tool is being STIRRED round rather than scrubbed along -> `stir-contents`.
- The brush is picked up in the same sub-task -> prefix
  `robot.pick(tool_name, side)`.

## Hand contract (derived)

```
close(<brace_side>)
```
One event: the brace. `goto_relative`, `rotate` and `move_arm` are all
contact-transparent, and the brush's grip never changes -- it is taken in an
earlier sub-task and kept. The brace hand is NOT released here; write that
where the video shows it let go.

## Known limits

- The stroke pattern is the same as `wipe-surface-back-and-forth`; the brace
  and the twist are what make this a separate entry.
- One stroke pair is written out; repeat the call for more strokes.
- `'inside'` and `'object_long'` are hardcoded: they are the maneuver's
  identity (down the bore, twisting about the brush's long axis).
