---
name: flip-object-over
description: Lift a held flat object clear, roll the wrist a half turn, release so it lands inverted, then grasp it again to pin it down.
---

# flip-object-over

```python
flip_object_over(robot, object_name, side, lift_delta, turn_deg=180.0)
```

## When to use

- A sub-task like "Turn over the <object>": gloves, calculators, napkins,
  pillows, hats, folders, clothes.
- Any flip where the object is ALREADY held and gravity finishes the motion —
  the `release` in the middle is the point of the maneuver, not a mistake.
- The trailing `grasp` is the re-pin: the object is flipped so the next
  sub-task can work on its other face.

## When NOT to use

- Turning a PAGE (the object stays put, the tool sweeps across it): a bare
  `servo_ee_delta` pair.
- Turning a threaded cap: `ratchet_twist_off`.
- Rotating a held object without letting go: the `rotate` primitive alone.
- Flipping something too big for one hand — no two-arm variant is encoded here.

## Gripper contract

```
open(<side>) -> close(<side>)
```
`servo_ee_delta` and `rotate` are gripper-transparent; the events are the
`release` (drop it inverted) and the re-`grasp` (pin it). Ends HELD.

## Known limits

- The release is open-loop — a tall or top-heavy object may not land flat, and
  the trailing `grasp` then closes on nothing.
- Page turning is NOT covered; it is a servo sweep with a tool in hand.
