---
name: tear-off-piece
description: Separate an attached piece from the sheet, roll or bag it is stuck to — pin the parent with one arm, pinch the piece with the other, pull along the tear line, and let the freed piece go. One slice; nothing is carried anywhere.
---

# tear-off-piece

```python
tear_off_piece(robot, piece_name, parent_name, tear_side, brace_side, tear_delta)
```

The whole of a `Tear off the <piece> from the <parent>` sub-task. There is no
`tear` primitive on the YAM surface, and the maneuver is not a `pick`: what
matters is that the piece is ATTACHED, so the pull has to be reacted somewhere.
The counter-hold is the identity — an adhesive seal or a perforation transmits
the whole tug straight into the parent, and a parent that is free to follow it
never separates.

## When to use

- A sub-task like `Tear off the <piece> ...` / `Tear the <piece> ...`:
  a label off a sticker sheet, an adhesive seal off a bag, a note off a pad,
  tape off a dispenser.
- The piece is held by adhesive or a perforation, and the parent is soft or
  light enough to travel with the pull.
- The freed piece is DROPPED where it lands. If it is then carried somewhere,
  that is the next sub-task.

## When NOT to use

- A part simply lifts OFF something with no counter-hold needed →
  `pull_part_free` (single-arm: `grasp` → pull → `release`, contract
  `close(<side>) -> open(<side>)`). Use that for "Remove the <part>", and this
  one for "Tear".
- The piece is being APPLIED to a target rather than separated → place it and
  `robot.attach(...)` inline.
- A zipper slider runs along a closure and nothing separates →
  `run_zipper_along_seam`.
- The piece is taken and carried away in the same sub-task → `pick`.

## Gripper contract

```
close(<brace_side>) -> close(<tear_side>) -> open(<tear_side>)
```

`grasp(parent, brace_side)` pins; `grasp(piece, tear_side)` bites; the pull is
gripper-transparent; `open_gripper(tear_side)` drops the freed piece. The
BRACE IS STILL HOLDING when the sub-task ends — the parent is under the arm
for whatever comes next.

## Known limits

- The contract `close(A) -> close(B) -> open(B)` is also what a brace plus any
  short-lived pinch produces, and it overlaps the opening of
  `run_zipper_along_seam`'s contract; the contract alone cannot tell a tear
  from a zipper run — only the sub-task label can.
- ONE pull is written out (skill bodies cannot loop). On a stiff perforation,
  call the skill again rather than widening `tear_delta` past what
  `servo_ee_delta` accepts.
- The pull is open-loop: no force sensing, so nothing detects that the piece
  actually came away. A too-small `tear_delta` silently leaves it attached.
- `tear_side` and `brace_side` must be different arms; the loader cannot
  enforce it.
- The parent is assumed reachable and graspable by name; a tear off a fixture
  that cannot be gripped has no realization here.
