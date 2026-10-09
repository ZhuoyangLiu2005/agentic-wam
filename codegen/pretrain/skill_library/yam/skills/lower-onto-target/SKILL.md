---
name: lower-onto-target
description: Detect the target, hover the carried object above it keeping the current wrist orientation, and lower it in - the standard placement motif. Gripper-transparent; the caller does the release.
---

# lower-onto-target

```python
lower_onto_target(robot, target_name, place_side, hover_height, lower_delta)
```

## When to use

- A "Place the <object> at/in/on the <target>" sub-task where the arm already
  holds the object and has to be driven to the target.
- The tell is the `detect_objects_oneshot -> get_robot_state -> move_arm`
  triple. `get_robot_state` is there for ONE reason - to reuse the arm's
  current rpy so the carried object is not rotated on the way over.
- Typical targets: containers, boxes, plates, cups, cases, racks, tubes.

## When NOT to use

- You want the whole semantic place including the open -> `robot.place(object,
  target, side)`, or `relocate_object` for the pick+place pair.
- The target is a graspable object you must get the gripper ONTO (to twist,
  press, or close on it) -> `approach_best_grasp`, which uses the grasp
  orientation instead of the current wrist orientation.
- The object has to be seated/pressed home -> `robot.attach(...)` or
  `brace_and_push_in`.
- The move is a small nudge from where the arm already is -> write the
  `servo_ee_delta` inline.

## Gripper contract

```
(gripper-transparent)
```

EMPTY on purpose. The release often belongs to the next sub-task, so pinning
an open here would be wrong. Compose `robot.release(place_side)` after the call
when the drop is part of your sub-task.

## Known limits

- Takes `detections[0]` (highest score) with no score gate; it misfires when
  several instances of the target are visible.
- `extent_3d` is ignored, so `hover_height` must be supplied by the caller
  rather than derived from the target's height.
- Shares its `detect -> get_robot_state -> move_arm` head with
  `scrub_back_and_forth`; the two differ in tail (one descent vs two opposed
  strokes) and in intent.
