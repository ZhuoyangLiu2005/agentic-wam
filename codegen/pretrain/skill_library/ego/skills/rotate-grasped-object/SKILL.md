---
name: rotate-grasped-object
description: Take an object where it sits and turn it about a semantic axis; the hand ends still on the object.
---

# rotate-grasped-object

```python
rotate_grasped_object(robot, object_name, side, axis, angle_deg)
```

## When to use

- A sub-task like "Rotate the `<X>`" / "Turn the `<X>` to face …" where the
  hand takes the object and turns it in the SAME sub-task. This is the grasp
  the `rotate` primitive deliberately does not include — `rotate` alone never
  changes the grip state.
- Works for an object turned IN the grip and for one spun in place on its
  support — `rotate` is defined over both.
- Typical values: `axis='object_symmetry'` (a phone, a cap) or `'vertical'`
  (a table, a machine, dough); `angle_deg=90.0`.

## When NOT to use

- The turn is a half-turn presenting the OTHER face, followed by bringing that
  face somewhere: `turn_over_object`.
- The object is a threaded part being unthreaded: `unscrew_cap`.
- The turn opens or closes a hinged closure: `open_closure` / `close_closure`.
- The turn empties a held vessel: `pour_into_target`.
- The object is ALREADY held from a previous sub-task: call the bare `rotate`
  primitive (this body would add a spurious re-grasp).

## Hand contract (derived)

    close(<side>)

`grasp` closes; `rotate` is contact-transparent. The hand ends STILL ON the
object — write the release in a later sub-task.

Contract collision: `pick_and_bring` derives the same single `close(<side>)`.
The bodies differ (grasp-in-place vs lift-and-carry); pick the one that
matches the video.

## Known limits

- A wrist twist on an already-touched object produces no visible aperture
  change, so the single `close` is hard to confirm from video alone.
- Two-handed rotates are NOT covered (both hands on the same object) — use
  `close_closure`'s shape or spell the calls out.
- `angle_deg` and `axis` are a semantic description of the motion, never
  measured.
