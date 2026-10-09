---
name: adjust-in-place
description: Grip an object where it lies, slide it a few centimetres without lifting, and let go - the "Adjust the <object>" sub-task.
---

# adjust-in-place

```python
adjust_in_place(robot, object_name, side, nudge_delta)
```

## When to use

- The sub-task is "Adjust the <object> ...", "Straighten ...",
  "Reposition ..." and the object stays on the surface.
- One arm, one small bounded translation (a few centimetres), gripper opens
  at the end.
- Typical objects: shoes, cables, keyboards, gloves, small bags or garments.

## When NOT to use

- The object is LIFTED and carried somewhere -> `relocate_object`.
- The thing being moved is a hinged/sliding CLOSURE being driven shut ->
  `push_closure_shut` (identical call shape, different sub-task; see the
  contract note below).
- Something is being pulled OPEN against a braced body ->
  `brace_and_pull_open`.
- Both arms hold the item and pull it taut -> `spread_taut_and_release`.
- The grip itself is what changes, not the pose -> `robot.regrasp(...)`.

## Gripper contract

```
close(<side>) -> open(<side>)
```

**Contract collision (declared):** identical to `push_closure_shut` and to
`relocate_object`. The three are distinguished by their bodies and their
sub-task, not by their contract. Pick by what the sub-task DOES.

## Known limits

- Single-arm only. Two-arm adjusts (both arms nudging a cloth) are
  `spread_taut_and_release`.
- ONE nudge per call; call again for a further correction.
- No re-grip: a `grasp -> regrasp` adjustment is not modelled.
