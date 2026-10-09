def brace_and_push_in(robot, object_name, brace_side, push_side, push_delta):
    """Brace the part low with one arm, take it higher up with the other, and
    push it home into its receptacle.

    The "Insert the <part> into the <receptacle>" sub-task. A long thin part --
    a straw, a plug, a document, a shoelace tip -- buckles if one arm pushes it
    alone, so it is gripped TWICE on the same object: the low grip keeps it
    aligned with the hole, the high grip supplies the push.

    Parameters (values below are EXAMPLES, not defaults baked in):
        object_name  the part being seated -- e.g. 'straw', 'plug', 'document'
        brace_side   arm that holds the part low and keeps it aligned
        push_side    arm that grips higher and pushes (the OTHER arm)
        push_delta   XYZ metres of the seating push, e.g. [0.0, 0.0, -0.05]
                     straight down into a lid hole, or along the hole axis

    Ends with BOTH grippers closed on the part, seated. Release whichever arm
    the next sub-task needs free.
    """
    # low grip: keeps the part aligned with the opening
    robot.grasp(object_name, brace_side)
    # high grip: the one that will do the pushing
    robot.grasp(object_name, push_side)
    # drive it home along the hole axis
    robot.servo_ee_delta(push_side, push_delta)
