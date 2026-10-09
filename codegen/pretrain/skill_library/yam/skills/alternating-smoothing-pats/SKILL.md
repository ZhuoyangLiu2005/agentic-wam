---
name: alternating-smoothing-pats
description: Press the fabric flat with one hand and come off, then the other hand — the alternating grasp/release pats of a "smooth out" sub-task.
---

# alternating-smoothing-pats

```python
alternating_smoothing_pats(robot, fabric_name, first_side, second_side)
```

## When to use

- A sub-task like "Smooth out the <sheet>": garments, posters, napkins,
  dishcloths, nets, blankets, pillows.
- Whenever the written form is a run of alternating `grasp` / `release` on the
  same object with no motion between them — each pair is one press-and-lift
  pat, and the alternation between arms is how the sheet gets covered.

## When NOT to use

- Dragging a hand ACROSS the cloth to push a wrinkle out (`grasp ->
  servo_ee_delta -> release`) — a different motion; write it inline.
- Folding the sheet afterwards -> `fold_edge_over`.
- Wiping a surface with a held cloth -> `scrub_back_and_forth`.
- The sheet must end PINNED for the next step — this skill ends with both
  hands off; add the `grasp` calls yourself.

## Gripper contract

```
close(<first_side>) -> open(<first_side>) -> close(<second_side>) -> open(<second_side>)
```
Two clean pats. Nothing is held before or after.

## Known limits

- Two pats per call (skill bodies cannot loop); call again for more.
- `grasp` is being used as a PRESS here — the API's `grasp` closes on a
  sampled grasp pose, so on a flat sheet it may pinch and lift the cloth
  rather than press it.
- Under a same-side binding the contract matches `hand_over_hand_extract`;
  the discriminator is the absence of any `servo_ee_delta`.
