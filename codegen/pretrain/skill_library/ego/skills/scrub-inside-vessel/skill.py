def scrub_inside_vessel(robot, vessel_name, tool_name, angle_deg, stroke_direction, return_direction, side, brace_side):
    """Clean the INSIDE of a vessel with a brush: the `brace_side` hand holds
    the vessel still, the `side` hand puts the brush down inside it and works
    it with a twist and a stroke pair -- the "Brush the <vessel>" subtask.

    The brace is what distinguishes this from wiping an open surface: a
    bottle, test tube or jar rolls away under the brush. The brace hand ENDS
    STILL ON the vessel (no release here) and the brush is already in the
    working hand. Example values: 'bottle', 'test tube', 'bottle brush';
    angle_deg=90.0, stroke/return = ('left','right') or ('up','down').
    """
    # Hold the vessel so it cannot roll or tip.
    robot.grasp(vessel_name, brace_side)
    # Put the brush down inside it.
    robot.goto_relative(side, vessel_name, 'inside')
    # Twist the bristles against the wall.
    robot.rotate(tool_name, 'object_long', angle_deg, side)
    # Work it up and back.
    robot.move_arm(side, stroke_direction)
    robot.move_arm(side, return_direction)
