def ratchet_twist_off(robot, part_name, side, turn_deg=-90.0):
    """Two ratchet bites on a threaded part: pinch, turn, let go, pinch, turn, let go.

    The SLICE-SIZED fragment of un/screwing, not a whole fastener maneuver.
    An annotated "Unscrew the <part> from the <receptacle>" sub-task is the
    bare ``close_gripper -> rotate -> open_gripper`` cycle and nothing else --
    the approach, the brace and carrying the freed part away are NEIGHBOURING
    sub-tasks. Two bites is the straight-line unroll of that cycle (skill bodies
    cannot loop); call it again for a deeper thread.

    Direction rides on the SIGN of ``turn_deg``, so the same body backs a part
    OUT (negative, the default) or seats one IN (positive).

    Parameters (values below are EXAMPLES, not defaults baked in):
        part_name  the part being turned -- e.g. 'lid', 'cap', 'knob', 'cover'
        side       arm that turns it, 'left' or 'right'
        turn_deg   wrist angle per bite; e.g. -90.0 to free a hand-tight lid,
                   +15.0 to twist a part down to seat

    The wrist is NOT unwound between bites (same-signed ``rotate`` every bite),
    which is why two bites is the practical limit of one call.
    """
    # bite 1 -- pinch the part, turn it, let go
    robot.close_gripper(side)
    robot.rotate(part_name, turn_deg, side)
    robot.open_gripper(side)
    # bite 2 -- ends OPEN, so the next slice can re-pinch, lift, or hand off
    robot.close_gripper(side)
    robot.rotate(part_name, turn_deg, side)
    robot.open_gripper(side)
