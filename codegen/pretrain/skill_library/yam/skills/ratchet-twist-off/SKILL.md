---
name: ratchet-twist-off
description: Two ratchet bites on a threaded part (pinch, turn, let go — twice); the slice-sized fragment of unscrewing, direction set by the sign of turn_deg.
---

# ratchet-twist-off

```python
ratchet_twist_off(robot, part_name, side, turn_deg=-90.0)
```

## When to use

- A sub-task like "Unscrew the <part> from the <receptacle>" — lids off jars
  and containers, battery covers, knobs, gears, caps.
- Also the SEATING twist at the end of an install ("twist the part in place
  until it is fully seated") — pass a small positive `turn_deg`.
- The part is already reachable and the wrist is already on it; this is the
  turning fragment only.

## When NOT to use

- The approach (`sample_grasp_pose` / `select_best_grasp` / `move_arm`) and the
  brace on the receptacle — those are their own sub-tasks.
- Carrying the freed part away: that is `pick` / `place` / `handover` in the
  calling program.
- A single unwinding turn with the grip kept: the `rotate` primitive alone.
- A lid that pops or pries straight off with no thread: `pull_part_free`.
- The wrist must UNWIND between bites (`open -> rotate(-x) -> close`): that is
  the neighbouring `ratchet_turn_fastener`, which carries a `backswing_deg`
  and ends mid-ratchet. This skill is the no-backswing form, and it ends OPEN.

## Gripper contract

```
close(<side>) -> open(<side>) -> close(<side>) -> open(<side>)
```

Ends OPEN — the part is left on its thread for the next sub-task to re-pinch,
lift, or hand over.

## Known limits

- The wrist is not unwound between bites; physically a real ratchet needs the
  unwind (`open -> rotate(-x) -> close -> rotate(+x)`, see
  `ratchet_turn_fastener`). Two bites is the practical limit before that
  matters.
- Nothing detects when the thread frees; the caller must decide when to stop
  calling.
- Contract `close -> open -> close -> open` is shared with
  `hand_over_hand_extract`; the discriminator is `rotate` vs `servo_ee_delta`
  between the gripper events.
