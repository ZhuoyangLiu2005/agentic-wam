---
name: flatten-by-pulling-taut
description: Pull a slack flat item taut against its support - one hand pins it, the other drags, re-bites further along, and drags again; the "Flatten out the <item>" subtask.
---

# flatten-by-pulling-taut

```python
flatten_by_pulling_taut(robot, item_name, direction, side, brace_side)
```

## When to use

- The sub-task reads "Flatten out the `<item>`" / "Spread open the
  `<item>`" and the frames show a garment, napkin, sheet or piece of fabric
  being DRAGGED flat -- with the hand letting go and re-biting further along
  mid-slice.
- The item is lying on a support (bed, table, machine bed) and stays there.
- The `regrasp` is what to look for: fingers releasing and re-closing on the
  same item without it moving.

## When NOT to use

- The item is PATTED flat, hand coming on and off it -> a `press` pair
  (written inline, or `press_onto_target`) -- a different contact contract
  entirely.
- Both hands take two corners and draw them apart in one go, then let go
  (`grasp; grasp; move_arm; move_arm; release; release`) -- that is the
  bimanual spread, byte-identical to `move_item_two_handed`'s body; write it
  inline or call that skill.
- The item is picked up and laid down first -> prefix `pick`/`place` before
  flattening.
- What follows is a fold -> `fold_over_and_crease`.

## Hand contract (derived)

```
close(<brace_side>) -> close(<side>) -> open(<side>) -> close(<side>)
```
Four events. The middle `open -> close` pair is `regrasp`'s declared effect --
the ego API's own name for a grip adjustment that leaves the object where it
is. Both hands end CLOSED; neither release is in this body.

## Known limits

- A `regrasp` on cloth -- fingers briefly opening while still touching fabric
  -- is hard to see on video, so the middle two contract events are the
  least visually verifiable.
- One re-bite is written out (skill bodies cannot loop); call again or add
  more `regrasp`/`move_arm` lines for a longer flatten.
- The support the item is flattened against is not a parameter (no call names
  it).
- `direction` is ego-frame and single; a real flatten often pulls two ways.
