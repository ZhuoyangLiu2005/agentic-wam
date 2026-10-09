---
name: run-zipper-along-seam
description: Pin the case with one arm, take the zipper pull with the other, settle the grip, draw it one run along the seam and let go.
---

# run-zipper-along-seam

```python
run_zipper_along_seam(robot, item_name, pull_side, brace_side, pull_delta)
```

## When to use

- A sub-task like "Zip the <case/bag/garment>": pencil cases, luggage, sports
  bags, storage cases, coats and jackets.
- Any zipper run where the body is soft and the pull tab is small — that is
  exactly why the brace and the regrasp are there.
- `pull_delta` carries the direction, so this also drives a zipper OPEN.

## When NOT to use

- Aligning the two halves of the zipper before it will run at all (both arms
  grasp the bottom of the coat) — that is a separate sub-task, written as two
  plain `grasp` calls.
- The case does not need pinning (heavy luggage on the floor): `grasp` the tab
  and `servo_ee_delta` directly.
- Closing a lid, latch or snap -> `push_closure_shut` or `robot.attach(...)`.

## Gripper contract

```
close(<brace_side>) -> close(<pull_side>) -> open(<pull_side>) -> close(<pull_side>) -> open(<pull_side>)
```

Brace closes first and NEVER opens — it is still holding when the skill
returns, ready for the next run down the same seam.

## Known limits

- One run per call; a full-length seam takes several.
- The pull tab has no separate detector name — the body grasps `item_name`
  twice and relies on the grasp sampler to find the tab.
- No two-arm alignment phase: if the zipper halves are not engaged, the run
  does nothing.
- `pull_delta` is caller-supplied and open-loop.
