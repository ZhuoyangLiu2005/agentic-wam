---
name: hang-item-on-support
description: Bring a held item over a hook, rack or branch and let it down onto it - the "Hang the <X> on the <Y>" subtask; the approach from above is what distinguishes it from a plain place.
---

# hang-item-on-support

```python
hang_item_on_support(robot, object_name, support_name, side)
```

## When to use

- The sub-task reads "Hang the `<X>` on/onto the `<Y>`" and the frames
  show an item with a loop, hook, handle or shoulder being brought OVER a
  rack, hook, holder, hanger or branch and released onto it.
- The item is already in the hand from an earlier slice.
- The support takes the whole load afterwards -- that is what makes it a hang
  rather than a rest.

## When NOT to use

- The item comes to rest ON a surface -> `place_at_target` or the bare `place`
  primitive; there is no over-and-down approach to encode.
- The item is pushed into a clip/clamp that grips it -> write
  `goto_relative(side, clamp, 'inside'); release(side)` inline.
- The item ends FIXED to the support by contact force -> `attach`.
- The hanger goes INSIDE the garment first (threading a hanger through a
  collar) -- write `goto_relative(..., 'inside')` plus a re-grasp inline.

## Hand contract (derived)

```
open(<side>)
```
One event: `place`'s declared let-go. `goto_relative` is contact-transparent.
The hand ENTERS holding the item and LEAVES empty.

## Known limits

- Two calls, one of which is a primitive: the skill earns its place only on
  the `'above'` approach. If the clearance move is not visible in the frames,
  call `place` directly.
- A hand opening around a thin hook is hard to read visually; the single open
  is a frame reading of the let-go.
- `'above'` is hardcoded; a side-loaded hook (a bag onto a peg) would need
  `'against'`, which this body cannot express.
- No clearance is given, so how far above the support the item is presented
  is unmodelled.
