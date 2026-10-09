def pull_part_free(robot, part_name, side, pull_delta):
    """Take hold of a part that is ON something and drag it straight off, then let go.

    The whole of a "Remove the <part> from the <thing>" sub-task: grip where
    it sits, one bounded pull along the axis it separates on, release. There is
    no lift and no transport -- disposing of the freed part ("...into the
    trash bin") is the NEXT sub-task, a plain ``place``.

    Parameters (values below are EXAMPLES, not defaults baked in):
        part_name   what comes off -- e.g. 'wrapper', 'battery cover', 'insole'
        side        arm that pulls, 'left' or 'right'
        pull_delta  XYZ metres of the pull, along the axis the part separates
                    on, e.g. [-0.15, 0.0, -0.1] to strip a wrapper off a straw

    The workpiece is often braced by the other arm first
    (``robot.grasp(<workpiece>, <other_side>)``) -- that brace is its own
    sub-task and stays out of this contract.
    """
    # grip the part where it sits, no lift
    robot.grasp(part_name, side)
    # one bounded pull along the separation axis
    robot.servo_ee_delta(side, pull_delta)
    # let it go -- transport is a separate sub-task
    robot.release(side)
