---
name: approach-best-grasp
description: Sample grasp candidates for an object, pick the best one reachable by this arm, drive the end-effector onto it and nudge in - the standard grasp-approach motif. Gripper-transparent.
---

# approach-best-grasp

```python
approach_best_grasp(robot, object_name, approach_side, close_in_delta)
```

## When to use

- Any sub-task whose first job is to GET THE ARM ONTO an object before
  doing anything with it: pick, screw, roll, insert, twist-in-place.
- The tell is the three-line `sample_grasp_pose / select_best_grasp /
  move_arm` triple, optionally followed by a small final approach.
- `close_in_delta` is that small final approach right after the `move_arm`.

## When NOT to use

- The target is a PLACE, not a graspable object (a board, a container, a
  drawer) -> `lower_onto_target`, which detects the target and holds the
  CURRENT wrist orientation instead of taking a grasp orientation.
- The object can simply be picked -> `robot.pick(...)` or `free_hand_and_pick`;
  this skill exists for the cases where the POSE itself is needed (to twist
  on it, to press, to close manually).
- No approach is needed because the arm is already there -> write the
  `servo_ee_delta` inline.

## Gripper contract

```
(gripper-transparent)
```

EMPTY, and that is the point. Perception + motion only, so this composes in
front of any contract without changing it.

## Known limits

- No gate on the grasp score.
- Assumes `sample_grasp_pose` returns at least one candidate.
- `top_down_only` / `camera` / `max_grasps` are left at their API defaults.
- Sibling of `free_hand_and_pick` (release-then-pick); choose this one when
  the grasp pose itself matters.
