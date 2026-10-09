---
name: stir-contents
description: Stir the contents of a vessel with the tool already held - drop the tool inside the vessel and sweep it round; the whole "Stir the <contents> in the <vessel>" subtask in one call.
---

# stir-contents

```python
stir_contents(robot, tool_name, vessel_name, side)
```

## When to use

- The sub-task is "Stir the <contents> in the <vessel>" and the hand ALREADY
  holds the stirring implement (rod, whisk, chopstick, egg beater) when the
  sub-task opens.
- One call = one whole stir sub-task. Repeat the call when the frames show a
  much longer agitation.

## When NOT to use

- The implement is picked up in this sub-task - that is its own sub-task
  (`pick`/`grasp`, or `pick-and-bring`); this skill assumes the grip already
  exists and emits no close.
- The tool leaves the vessel in this sub-task - add the exit line yourself,
  e.g. `goto_relative(side, vessel_name, 'above')` or `move_arm(side, 'up')`.
  Often the sub-task simply ends with the tool still in the vessel.
- The VESSEL is swirled by the hand holding it, or a held object is turned
  about its own axis - write `rotate` directly.

## Hand contract (derived)

```
(gripper-transparent - no close/open events)
```
`goto_relative` and `rotate` are contact-transparent: the hand enters the
sub-task already closed on the tool and leaves it still closed. This skill
makes NO contact claim - deliberately, because the grip is established
earlier.

## Known limits

- The sweep count is fixed by the body (two full sweeps, since skill bodies
  cannot loop), so the duration fidelity of a single call is approximate.
- `axis='vertical'` and `angle_deg=360.0` are literals; a different stirring
  axis or a partial sweep needs raw `rotate` calls.
- The vessel is not braced; a caller can add `grasp(vessel_name, other_side)`
  (see `brace-and-take-tool`).
