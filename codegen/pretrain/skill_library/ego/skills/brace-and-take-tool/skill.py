def brace_and_take_tool(robot, workpiece_name, brace_side, tool_name, working_side):
    """Set up the two-handed working stance: the brace hand closes on the
    workpiece where it sits to steady it, and the working hand closes on the
    tool it is about to use. Both hands keep holding.

    This is the whole of a "Hold the <workpiece> (securely)" subtask: the
    annotation names only the braced object, but the slice also contains the
    working hand taking its tool. Nothing is lifted and nothing is let go -
    the release belongs to whichever later subtask ends the brace.

    For a two-handed brace on ONE object with no tool (both hands steadying a
    bowl), pass the same name for `workpiece_name` and `tool_name`.
    """
    # Brace the workpiece where it sits: steadied, not carried.
    robot.grasp(workpiece_name, brace_side)
    # The other hand closes on the tool it is about to work with.
    robot.grasp(tool_name, working_side)
