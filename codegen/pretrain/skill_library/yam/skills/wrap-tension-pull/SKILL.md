---
name: wrap-tension-pull
description: Two hand-over-hand bites on a strand being wound — regrasp further along, pull the slack taut, twice.
---

# wrap-tension-pull

```python
wrap_tension_pull(robot, strand_name, side, pull_delta)
```

## When to use

- A sub-task like "Wrap the <strand> around the <core>" where the strand is
  already in hand: shoelaces, cables onto a reel, tape onto a pipe, an elastic
  band around a stack.
- Any winding sub-task written as an alternation of `regrasp` and a short
  pull — the gripper cannot follow a full turn, so the wrap advances one bite
  at a time.

## When NOT to use

- The wrap advances by TURNING the core instead of pulling the strand
  (`open -> rotate -> close -> rotate`): use `ratchet_turn_fastener` on the
  core, or the `rotate` primitive.
- Tightening a tie or knot rather than laying turns: write the opposed
  `servo_ee_delta` pulls inline.
- The strand is not yet held: `grasp` / `pick` first.

## Gripper contract

```
open(<side>) -> close(<side>) -> open(<side>) -> close(<side>)
```

Two `regrasp` expansions. Ends HELD — the strand stays captive so the next call
can lay the next turn.

## Known limits

- Two bites per call (skill bodies cannot loop); call again for more.
- The rotate-the-core realization of wrapping is not modelled here.
- The core is normally braced by the other arm; that brace is a separate
  call and stays out of this contract.
- `regrasp` re-grips at the SAME place semantically — advancing along the
  strand is an assumption of this skill, not something the API guarantees.
