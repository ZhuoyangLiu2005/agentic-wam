def ratchet_turn_fastener(robot, fastener_name, turn_side, angle_deg, backswing_deg):
    """Two ratchet strokes on a threaded fastener: grip, turn, LET GO, swing the
    wrist back, grip again, turn again.

    A wrist can only rotate so far, so screwing is spelled as a ratchet: the
    open in the middle is the release that lets the wrist unwind WITHOUT
    unscrewing what was just turned. Two strokes are written out because skill
    bodies cannot loop; call again for more strokes.

    Parameters (values below are EXAMPLES, not defaults baked in):
        fastener_name  what turns -- e.g. 'lid', 'knob', 'nut', 'bulb', 'cap'
        turn_side      arm that turns, 'left' or 'right'
        angle_deg      degrees of ONE driving stroke, e.g. 90.0
        backswing_deg  degrees of the open-jaw return, normally the NEGATION of
                       angle_deg, e.g. -90.0

    Ends CLOSED on the fastener, mid-turn -- call again for more strokes, or
    `robot.release(turn_side)` when it is seated.

    SINGLE-ARM ON PURPOSE. Bracing the body being screwed into is real and
    common but its close is not part of the ratchet train, so it stays out of
    this contract: write `robot.grasp(body_name, other_side)` before the call.
    """
    # stroke 1: grip and drive
    robot.close_gripper(turn_side)
    robot.rotate(fastener_name, angle_deg, turn_side)
    # let go and unwind the wrist without unscrewing
    robot.open_gripper(turn_side)
    robot.rotate(fastener_name, backswing_deg, turn_side)
    # stroke 2: grip again and drive again
    robot.close_gripper(turn_side)
    robot.rotate(fastener_name, angle_deg, turn_side)
