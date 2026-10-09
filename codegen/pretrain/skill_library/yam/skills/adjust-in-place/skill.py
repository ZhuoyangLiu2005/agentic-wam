def adjust_in_place(robot, object_name, side, nudge_delta):
    """Nudge an object a few centimetres where it lies, then let go.

    The "Adjust the <object>" sub-task: grip it where it sits,
    slide/straighten it by a small bounded delta WITHOUT lifting, and release.
    The object stays on the surface the whole time.

    Parameters (values below are EXAMPLES, not defaults baked in):
        object_name  what is nudged -- e.g. 'shoe', 'cable', 'keyboard'
        side         arm that does it, 'left' or 'right'
        nudge_delta  XYZ metres of the nudge, e.g. [0.0, -0.05, 0.0] or
                     [-0.05, 0.0, 0.0] (a few centimetres)

    Ends with the gripper OPEN and the object resting in its new pose.
    """
    # take hold where it sits -- no lift
    robot.grasp(object_name, side)
    # slide it by a small bounded amount
    robot.servo_ee_delta(side, nudge_delta)
    # let it settle
    robot.release(side)
