def roll_up_one_bite(robot, roll_side, roll_delta, return_delta):
    """One bite of rolling something up: pinch, push the roll forward, let go,
    come back for the next bite.

    Rolling is reach-limited -- the arm can only push the roll so far before it
    has to release, travel back to where the material now is, and pinch again.
    That re-bite is the whole sub-task: the open in the middle is NOT a drop,
    the roll stays rolled because it is resting on itself.

    Parameters (values below are EXAMPLES, not defaults baked in):
        roll_side     arm that rolls, 'left' or 'right' (arms may also
                      alternate bite by bite across calls)
        roll_delta    XYZ metres of ONE push, e.g. [0.0, 0.05, 0.0]
        return_delta  XYZ metres back to the new leading edge -- normally the
                      negation of roll_delta, e.g. [0.0, -0.05, 0.0]

    Ends with the gripper OPEN, parked at the next pinch point.
    """
    # pinch the leading edge of the roll
    robot.close_gripper(roll_side)
    # push the roll forward by one bite
    robot.servo_ee_delta(roll_side, roll_delta)
    # let go -- the roll stays rolled
    robot.open_gripper(roll_side)
    # travel back to where the material now leaves the roll
    robot.servo_ee_delta(roll_side, return_delta)
