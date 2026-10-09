---
name: wipe-surface-back-and-forth
description: Work a cloth or sponge already in one hand over an open surface - park the hand on it, then two out-and-back strokes; the "Wipe the <surface>" subtask.
---

# wipe-surface-back-and-forth

```python
wipe_surface_back_and_forth(robot, surface_name, relation, stroke_direction, return_direction, side)
```

## When to use

- The sub-task is "Wipe the `<X>`" / "Wipe off the `<X>`" / "Wipe the residue
  from the `<X>`" and the frames show a held cloth, sponge or brush swept
  repeatedly across an OPEN, flat-ish surface.
- The wiping tool is already in the hand from an earlier sub-task.
- Typical `relation`: `'above'` (tray, plate), `'against'` (a face being
  scrubbed). Typical stroke pairs: `('left','right')` and
  `('toward_body','away_from_body')`.
- Wrap in a program-level `for _ in range(n):` for a long scrub.

## When NOT to use

- The tool goes INSIDE a vessel and is twisted -> `scrub-inside-vessel`
  (that one braces the vessel and adds the `rotate`).
- The hand PATS the item flat rather than sweeping it -> `press-onto-target`
  (a `press` pair; a different contact contract).
- A substance is being pushed around to cover a surface ("Spread the batter /
  the jam") -- same shape; call this skill and say so in the comment, there
  is no separate spread skill.
- The cloth is picked up in the same sub-task -> add
  `robot.grasp(cloth_name, side)` before the call.

## Hand contract (derived)

```
(gripper-transparent)
```
EMPTY on purpose. `goto_relative` and `move_arm` are both contact-transparent
in the ego API, so this skill makes no claim about the hand at all -- it rides
along checkably-neutral on whatever grip the previous sub-task left.

## Known limits

- Two stroke pairs are written out; repeat the call for more strokes.
- The skill does not name the wiping TOOL. Name it in a comment at the call
  site.
- `move_arm` distance is left unset; the stroke length is unmodelled.
