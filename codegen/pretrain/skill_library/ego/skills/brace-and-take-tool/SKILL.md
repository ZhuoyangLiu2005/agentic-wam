---
name: brace-and-take-tool
description: Set up the two-handed working stance - the brace hand closes on the workpiece where it sits, the working hand closes on the tool it is about to use; the whole "Hold the <X> securely" subtask, nothing lifted and nothing let go.
---

# brace-and-take-tool

```python
brace_and_take_tool(robot, workpiece_name, brace_side, tool_name, working_side)
```

## When to use

- The subtask reads "Hold the <X>", "Hold the <X> securely" or "Secure
  the <X>" and the frames show one hand steadying the workpiece in place while
  the other closes on the implement it will use next (knife, funnel, spoon).
- Both hands END HOLDING. The slice is a STANCE, not an action: it exists so
  the next subtask can act.
- A two-handed brace on ONE object with no tool (both hands steadying a
  bowl) - pass the same name for `workpiece_name` and `tool_name`.

## When NOT to use

- The second hand LIFTS its object off/out of the workpiece - use
  `brace_and_pick` (`grasp(support)` + `pick(object)`). The distinction is the
  API's own: `grasp` closes on something the hand will NOT carry, `pick`
  carries it away. `brace_and_take_tool` is the no-lift twin.
- Only ONE hand is involved - the ego API has no `hold` verb by design; write
  `robot.grasp(object_name, side)` with an intent comment.
- The working hand's action (a cut stroke, a pour, an insert) is IN THE SAME
  slice - write the brace inline and then the action; this skill stops at the
  stance so it stays one slice.
- The steadying hand holds the object ABOVE its support (carrying it) - that
  is `pick`, not a brace.
- BOTH hands then carry the item somewhere and let go - use
  `move_item_two_handed`, whose first two lines are the same `grasp` pair
  but whose identity is the carry and the two releases. This skill is the
  stance that goes NOWHERE.

## Hand contract (derived)

```
close(<brace_side>)
close(<working_side>)
```
Two closes, no opens - by design. The brace outlives the slice; any program
using this skill must account for both releases in later slices, where the
video shows each hand leaving.

## Known limits

- The workpiece/tool asymmetry is an interpretation: structurally the body is
  two `grasp` calls, and nothing in the trace proves which hand is the brace.
- Annotations phrased "Secure the <X>" describe the same stance and are
  covered by this skill.
- Do not confuse with `brace_and_pick`: if the second object is visibly
  carried away, that is the other skill.
