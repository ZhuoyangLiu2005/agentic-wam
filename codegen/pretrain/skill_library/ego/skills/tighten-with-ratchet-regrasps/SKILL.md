---
name: tighten-with-ratchet-regrasps
description: Take up the last of a thread on an already-seated cap or fitting - reach onto it, turn, release the bite, re-bite and turn again, ending with the fingers open on the part.
---

# tighten-with-ratchet-regrasps

```python
tighten_with_ratchet_regrasps(robot, object_name, relation, axis, angle_deg, side)
```

## When to use

- The sub-task is "Tighten the `<X>` on/onto the `<Y>`" and the frames show a
  cap/fitting that is ALREADY on its thread being ratcheted the last part of
  a turn: turn, fingers slip back, turn again.
- The other hand is usually holding the body steady. That brace is the
  caller's `robot.grasp(body_name, other_side)` line -- it outlives this
  skill.
- Typical values: `relation='above'`, `axis='object_symmetry'`,
  `angle_deg=90.0` (sign = tightening handedness).

## When NOT to use

- The part is being threaded on from loose -> write the bite/turn sequence
  with primitives (that bites FIRST; this one turns first because the hand is
  already on the part).
- The part is coming off -> `unscrew-cap`.
- "Tighten the drawstring / zip tie / lace" -- a cord pulled taut, not a
  thread turned. Use `pull-open-and-hold` per hand or write the two-hand
  `grasp`/`move_arm` pull inline.
- A single final nip with no release -> the `rotate` primitive.

## Hand contract (derived)

```
open(<side>) -> close(<side>) -> open(<side>)
```
The hand ENTERS closed on the part (that is what "already seated and held"
means) and LEAVES open. `goto_relative` and `rotate` are contact-transparent.
Note the contract STARTS with an open -- the phase difference from screwing a
part on from loose.

## Known limits

- A ratcheting hand opens and closes rapidly on a small cap; the three
  declared events summarize a run that is hard to resolve event-for-event
  from video.
- The boundary with screwing on is a phase difference read off the sub-task
  verb, not off the frames.
- Two bites are written out; repeat the call for more.
- The `goto_relative(..., relation)` approach is a motion scaffold; drop it
  (call `rotate` directly) when the hand is already there.
