def alternating_smoothing_pats(robot, fabric_name, first_side, second_side):
    """Press and let go with one hand, then the other -- flattening pats on cloth.

    The "Smooth out the <garment/sheet>" sub-task: NO transport, NO dragging
    across the cloth, just alternating grasp/release pats that push the
    fabric down flat. Each ``grasp`` is a press where the fabric lies and each
    ``release`` is the hand coming off again; the alternation between arms is
    what covers the sheet.

    Parameters (values below are EXAMPLES, not defaults baked in):
        fabric_name  the sheet being flattened -- e.g. 'napkin', 'blanket',
                     'poster'
        first_side   arm that pats first, 'left' or 'right'
        second_side  the other arm

    ENDS with both hands OFF the fabric. If the next step needs the sheet
    pinned, issue those ``grasp`` calls in the calling program.
    """
    # pat 1 -- press the fabric flat, then come off
    robot.grasp(fabric_name, first_side)
    robot.release(first_side)
    # pat 2 -- the other hand, on the other half of the sheet
    robot.grasp(fabric_name, second_side)
    robot.release(second_side)
