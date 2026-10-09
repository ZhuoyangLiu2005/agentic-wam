---
name: open-closure
description: Swing a hinged closure open with one hand while the other braces the container body, then come off the closure.
---

# open-closure

```python
open_closure(robot, container_name, closure_name, side, brace_side, axis, angle_deg)
```

## When to use

- A subtask like "Open the `<lid|cover|door|flap|clip|switch>` (on|of)
  `<container>`" where the closure stays ATTACHED and swings about a hinge,
  and the video shows the other hand holding the container body.
- Typical values: `axis='object_long'` (a book cover) or `'lateral'` (a lid
  hinged away from the body); `angle_deg=+90.0` or `+180.0` — the SIGN is the
  opening sense. Example pairs: ('book', 'book cover'), ('case', 'case lid'),
  ('tray', 'tray lid'), ('pill box', 'pill box lid'), ('machine', 'switch').
- The brace OUTLIVES the skill. Write `robot.release(brace_side)` yourself
  where the video shows that hand leaving (usually a later subtask).

## When NOT to use

- The closure comes OFF the container: a screw cap is `unscrew_cap`; a
  lift-off/pull-off part is `pull_part_free`.
- The closure SLIDES or PULLS open (a drawer, a bag mouth, a zip): write
  `grasp -> move_arm -> release` out, or use `pull_open_and_hold` if the hand
  keeps holding; it is not this skill.
- BOTH hands work the same cover: that is `close_closure`'s shape with the
  opening angle sign.
- Closing the same closure again: `close_closure`.
- Nothing is braced and nothing is let go: `rotate_grasped_object`.

## Hand contract (derived)

    close(<brace_side>) -> close(<side>) -> open(<side>)

`grasp` closes; `release` opens. The `brace_side` hand closes and STAYS
closed — this skill emits no open for it.

## Known limits

- The contract is a code-level claim; a hinge-open on a small cap may show no
  visible change in hand aperture.
- "Open" has two genuinely different realizations and this skill covers only
  the swing one; the pull/slide family (bags, drawers, zips) is left to
  explicit code.
- **The brace is not optional in this body.** For an unbraced hinge-open,
  spell `grasp`/`rotate`/`release` out rather than passing a fake
  `brace_side`.
- `angle_deg` is a semantic swing amount, not a measured one.
