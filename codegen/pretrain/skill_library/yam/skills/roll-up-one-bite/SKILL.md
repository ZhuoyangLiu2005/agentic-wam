---
name: roll-up-one-bite
description: One reach-limited bite of rolling: pinch, push the roll forward, let go, travel back for the next pinch. Ends open at the next pinch point.
---

# roll-up-one-bite

```python
roll_up_one_bite(robot, roll_side, roll_delta, return_delta)
```

## When to use

- A "Roll the <item>" / "Roll up ..." sub-task: a napkin, a t-shirt, a sock, a
  roller shade, a blanket, thread wound into a ball.
- The tell is the mid-push OPEN: the gripper lets go while the item stays
  rolled, then travels back. `close -> servo -> open -> servo` is the unit;
  repeat the call for as many bites as the item needs.

## When NOT to use

- A long object is being drawn OUT of a bag, sleeve or slot ->
  `hand_over_hand_extract`.
- A strand is being wound around something under tension ->
  `wrap_tension_pull`.
- The roll is being twisted in hand about the gripper axis rather than pushed
  along the surface -> `ratchet_turn_fastener` is the twist unit
  (`close / rotate / open`), or write `robot.rotate` inline.
- The cloth is being folded, not wound -> `fold_edge_over`.
- One short push with no re-bite: write the `servo_ee_delta` inline.

## Gripper contract

```
close(<roll_side>) -> open(<roll_side>)
```

Single-arm bite. Ends OPEN on purpose so the next bite (or the next sub-task)
can start from the pinch. Both `servo_ee_delta` phases are gripper-transparent.

## Known limits

- ONE bite per call (skill bodies cannot loop); call again for more.
- Single-arm. An alternating-arm roll (left bite, right bite, ...) needs
  successive calls with different `roll_side`.
- The in-hand TWIST realization of rolling (`close / rotate / open`) is not
  modelled here.
- Both deltas are caller-supplied and open-loop.
