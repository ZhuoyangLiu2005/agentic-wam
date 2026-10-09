---
name: scoop-and-carry
description: Take a load out of a source container with a held utensil and bring it over the destination - down inside, straight up, across and above; the "Scoop a <portion> from the <source>" subtask.
---

# scoop-and-carry

```python
scoop_and_carry(robot, source_name, target_name, side)
```

## When to use

- The sub-task is "Scoop (up) a `<portion>` from the `<source>`" and the
  frames show a spoon/scoop/spatula going into a pot, cooker, bowl or
  container, coming out loaded, and being carried over a plate/bowl/container.
- The utensil is already in the hand (it is not re-grasped inside a scoop).
- Use it for a single scoop; portioning repeats it as further sub-tasks.

## When NOT to use

- The load is tipped out of a VESSEL rather than lifted on a utensil ->
  `pour-into-target`.
- The utensil is being stirred round in the vessel -> `stir-contents`.
- The utensil is picked up in the same sub-task -> prefix
  `robot.pick(utensil, side)` or use `pick-and-bring`.
- The sub-task ends with the load already delivered -> add the delivery line
  (`rotate` to tip, `move_arm('down')`, or `place`); this skill stops with the
  utensil hanging above the target.

## Hand contract (derived)

```
(gripper-transparent)
```
EMPTY on purpose: `goto_relative` and `move_arm` are contact-transparent, and
the utensil's grip never changes during a scoop.

## Known limits

- The empty contract means the skill says nothing about the hand.
- The **loading tilt is deliberately not in the body**: add
  `rotate(utensil,'lateral',45.0,side)` between the dip and the lift when the
  video shows the wrist turn -- it is contact-transparent and rides along
  neutral.
- The utensil is not a parameter and neither is the material being scooped.
- `'inside'` and `'above'` are hardcoded because they ARE the maneuver's
  identity; a scoop that goes `'against'` a surface is a scrape, not this.
