def spread_taut_and_release(robot, left_delta, right_delta):
    """Both arms already hold the item: pull it taut in opposite directions,
    then let go -- the flatten / smooth-out sub-task.

    "Flatten out the <garment>" sub-tasks are built out of this: with a grip
    already established on both sides, drag the two grippers apart so the
    wrinkles come out, then release both so the item settles flat. Repeat the
    call (after re-gripping) for a thorough flatten.

    Parameters (values below are EXAMPLES, not defaults baked in):
        left_delta   XYZ metres the LEFT gripper drags, e.g. [0.0, 0.1, 0.0]
                     or [-0.1, 0.0, 0.0]
        right_delta  XYZ metres for the RIGHT gripper, normally the NEGATION of
                     left_delta, e.g. [0.0, -0.1, 0.0] / [0.1, 0.0, 0.0]

    Takes no object name on purpose: nothing here names the item -- the grip is
    already established by the caller (`robot.grasp(item, 'left')` /
    `robot.grasp(item, 'right')`, or a preceding `fold_edge_over`).

    BIMANUAL BY CONSTRUCTION: the arms are the literals 'left' and 'right'.
    """
    # drag the two grips apart so the wrinkles come out
    robot.servo_ee_delta('left', left_delta)
    robot.servo_ee_delta('right', right_delta)
    # let the item settle flat
    robot.release('left')
    robot.release('right')
