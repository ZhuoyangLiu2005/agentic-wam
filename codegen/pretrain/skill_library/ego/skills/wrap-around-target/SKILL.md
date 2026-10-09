---
name: wrap-around-target
description: Lead a held wrapper (ribbon, foil, film, gauze) once around a target in a closed circuit - over the top, down the far face, back underneath, and up.
---

# wrap-around-target

```python
wrap_around_target(robot, target_name, side)
```

## When to use

- The sub-task is "Wrap the `<wrapper>` around the `<target>`" and the frames
  show one hand carrying the wrapper in a CIRCUIT around a box, plate, bundle
  or limb.
- The wrapper is already in the hand from an earlier sub-task.
- Repeat with `for _ in range(n):` for a multi-turn wrap; follow with the tuck
  (`press`), the tie, or the `release` the video shows.

## When NOT to use

- Both hands take two edges and draw them over the item (foil/film folded onto
  a plate) -- that is a bimanual fold; write it inline or use
  `fold-over-and-crease`.
- The wrapper is TWISTED about its own axis rather than led around the target
  (a hair tie twisted onto a ponytail) -> `rotate` in a literal loop.
- The circuit is not closed (a single sweep of film pulled across) ->
  `move_arm` alone.

## Hand contract (derived)

```
(gripper-transparent)
```
EMPTY on purpose: `goto_relative` and `move_arm` are contact-transparent, and
a wrap never changes the grip on the wrapper.

## Known limits

- Wrapping is realized in several ways (circuit / bimanual edge-fold /
  twist-in-place); this body covers only the circuit.
- The empty contract means the skill says nothing about the hand.
- The four directions are ego-frame and hardcoded; a target approached from
  the side would need the circuit rotated, which this body cannot express.
- The wrapper is not a parameter.
- `move_arm` distances are unset, so the circuit's scale is unmodelled.
