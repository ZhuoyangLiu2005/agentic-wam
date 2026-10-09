---
name: pull-part-free
description: Brace an assembly with one hand and pull a part off/out of it with the other along a semantic direction.
---

# pull-part-free

```python
pull_part_free(robot, part_name, assembly_name, side, brace_side, direction)
```

## When to use

- A sub-task like "Remove the `<X>` from the `<Y>`" / "Pull the `<X>` off"
  for a part that comes away by a STRAIGHT PULL — a friction-fit cap, a key
  out of a lock, a nipple out of a ring, a plug out of a nozzle.
- The assembly needs holding so it does not come along with the part.
- Example pairs: ('end cap', 'knife handle'), ('key', 'lock'),
  ('battery', 'battery cover'), ('plug', 'nozzle'). Example
  `direction='right'` or `'toward_body'`.

## When NOT to use

- The part LIFTS clear rather than pulling off: `brace_and_pick`.
- The part unthreads: `unscrew_cap`.
- The closure stays attached and swings: `open_closure`.
- Both hands pull the two pieces APART in opposite directions (a balloon off a
  pump): write the two `move_arm` calls yourself; this body pulls with one
  hand only.
- Nothing needs bracing: use the bare `grasp` + `move_arm`.

## Hand contract (derived)

    close(<brace_side>) -> close(<side>)

Both hands end CLOSED — the part is in the `side` hand and the brace is still
on the assembly. Write both releases where the video shows those hands
leaving.

Contract collision: `brace_and_pick` derives the same
`close(<brace_side>) -> close(<side>)`. Choose by the motion: a directed PULL
here (`move_arm`), a LIFT clear there (`pick`).

## Known limits

- Removals are realized in many ways (pull, lift, two-handed separation); this
  body covers only the braced one-hand straight pull.
- No releases: a sub-task that also puts the removed part down is not this
  skill.
- `move_arm` distance is unset; how far the part travels is unmodelled.
