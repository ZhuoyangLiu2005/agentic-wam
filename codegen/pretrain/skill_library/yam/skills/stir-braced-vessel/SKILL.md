---
name: stir-braced-vessel
description: Stir inside a braced vessel with a stirrer the tool arm already holds — take the cup or bowl, drop the tip past the rim, RE-SEAT the brace now that the tip loads the wall, then walk one out-and-back sweep. The brace is still holding at the end.
---

# stir-braced-vessel

```python
stir_braced_vessel(robot, vessel_name, brace_side, tool_side,
                   plunge_delta, sweep_delta, return_delta)
```

The whole of a `Stir the <contents> in the <vessel>` sub-task. There is no
`stir` primitive on the YAM surface; what this adds over a run of
`servo_ee_delta` is the BIMANUAL BRACE AND ITS RE-SEAT: an unheld cup walks
across the table the moment a spoon loads its wall, and a vessel first taken
at the rim rolls in the fingers, so the brace arm takes it and then takes it
again once the tip is down.

## When to use

- A sub-task like `Stir the <contents> in the <vessel>` /
  `Stir the <contents> in the <vessel> with the <tool>`: liquid in a cup,
  cereal in a bowl, soup in a pan.
- Whenever a held tool circulates INSIDE a vessel wall rather than travelling
  over an open surface.
- The `tool_side` arm must ALREADY hold the stirrer — that is why no stirrer
  name is in the signature. Fetch the spoon or stick with a preceding
  `grasp`/`pick`; it is never let go here.

## When NOT to use

- A held tool or cloth is dragged over an OPEN surface → `scrub_back_and_forth`.
- The vessel is only being held steady while the other arm does something
  else → bare `grasp` (never wrap a primitive).
- The contents are tipped out → `tilt_pour_over_target`.
- The vessel is picked up and carried → `pick`.

## Gripper contract

```
close(<brace_side>) -> open(<brace_side>) -> close(<brace_side>)
```

`grasp(vessel, brace_side)` then `regrasp(vessel, brace_side)`. The STIRRING
arm produces NO gripper event at all — the stirrer is held throughout — so
`tool_side` never appears in the contract. The brace is STILL HOLDING when
the sub-task ends.

## Known limits

- The contract `close(A) -> open(A) -> close(A)` is what any `grasp` followed
  by a `regrasp` on the same arm produces, so it does not by itself identify
  a stir.
- `tool_side` appears in no gripper effect, so the checker cannot enforce that
  the stirring arm and the brace arm differ, nor that the stirrer is held at
  all. An empty `tool_side` gripper silently stirs air.
- One sweep pair is written out (skill bodies cannot loop). Call the skill
  again for a longer or thicker mix.
- The plunge is open-loop: `plunge_delta` is commanded blind with no force or
  liquid-level sensing, so a too-large value drives the stirrer into the
  vessel floor.
- The sweep is a diagonal out-and-back, not a true circle; a circular path
  would need more legs.
