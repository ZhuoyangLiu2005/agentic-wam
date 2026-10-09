def run_zipper_along_seam(robot, item_name, pull_side, brace_side, pull_delta):
    """Pin the case, take the zipper pull, settle the grip, draw it along the seam.

    The "Zip the <case/bag/garment>" sub-task. A zipper pull is a small hard tab
    on a soft body: the case MUST be pinned or it travels with the pull, and
    the first grasp on the tab is usually poor, so the pull arm regrasps before
    running it. One run per call -- a long seam takes several calls.

    Direction rides on ``pull_delta``, so the same body closes a zipper or
    opens one.

    Parameters (values below are EXAMPLES, not defaults baked in):
        item_name   the zipped item -- e.g. 'pencil case', 'bag', 'jacket'
        pull_side   arm that drives the zipper, e.g. 'right'
        brace_side  arm that pins the case flat, e.g. 'left'
        pull_delta  XYZ metres of the run along the seam, e.g.
                    [0.0, -0.1, 0.0] or [0.15, 0.0, 0.0]

    ENDS with pull_side OPEN (the tab is let go at the end of the run) and
    brace_side still holding, ready for the next run.
    """
    # pin the case so it does not travel with the zipper
    robot.grasp(item_name, brace_side)
    # take the zipper pull
    robot.grasp(item_name, pull_side)
    # the first bite on a small tab is usually poor -- settle it
    robot.regrasp(item_name, pull_side)
    # run the zipper along the seam
    robot.servo_ee_delta(pull_side, pull_delta)
    # let the tab go; the case stays pinned
    robot.release(pull_side)
