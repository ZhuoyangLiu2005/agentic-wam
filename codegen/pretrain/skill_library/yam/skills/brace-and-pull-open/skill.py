def brace_and_pull_open(robot, anchor_name, handle_name, brace_side, open_side, open_delta):
    """Hold the body of a thing still with one arm, take its lid / cap / flap /
    handle with the other, and draw it open.

    The "Open the <container>" sub-task. A one-armed pull just drags the whole
    object across the table, so the brace comes first: `grasp -> grasp ->
    servo_ee_delta` is the shape, and the brace lets go at the end because the
    opened part is now held by the other arm.

    Parameters (values below are EXAMPLES, not defaults baked in):
        anchor_name  the body being held still -- e.g. 'storage box',
                     'paper bag', 'notebook'
        handle_name  the part being drawn open -- e.g. 'cap', 'lid',
                     'drawer handle', 'cover'. May name the SAME physical
                     object as anchor_name when a bag mouth is pulled apart.
        brace_side   arm that holds the body still
        open_side    arm that draws the part open (must be the OTHER arm)
        open_delta   XYZ metres of the opening travel, e.g.
                     [-0.05, 0.0, 0.05] (pull a cap off) or
                     [0.0, 0.15, -0.1] (swing a cover over)

    Ends with `open_side` HOLDING the opened part and `brace_side` free.
    """
    # hold the body still so the pull does not drag the whole object
    robot.grasp(anchor_name, brace_side)
    # take the lid / cap / flap / handle
    robot.grasp(handle_name, open_side)
    # draw it open
    robot.servo_ee_delta(open_side, open_delta)
    # the brace is done; the opened part is held by the other arm
    robot.release(brace_side)
