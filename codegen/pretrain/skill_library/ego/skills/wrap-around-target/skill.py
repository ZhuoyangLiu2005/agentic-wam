def wrap_around_target(robot, target_name, side):
    """Lead the wrapper ALREADY HELD by the `side` hand once around a target:
    over the top, down the far face, back underneath toward the body, and up
    -- the "Wrap the <wrapper> around the <target>" subtask.

    ONE circuit per call; write a multi-turn wrap as a literal `for` loop over
    this call. Contact-transparent: the ribbon/foil/film/gauze is taken in an
    earlier subtask and is still held afterwards, so the tuck or the let-go at
    the end is the caller's line. The four directions are hardcoded because
    the CLOSED CIRCUIT is the maneuver's identity -- a wrap that does not come
    back where it started is a `move_arm`, not this. Example wrappers:
    'ribbon', 'foil', 'plastic wrap', 'gauze'.
    """
    # Start over the top of the target.
    robot.goto_relative(side, target_name, 'above')
    # Over the far side.
    robot.move_arm(side, 'away_from_body')
    # Down its far face.
    robot.move_arm(side, 'down')
    # Back underneath, toward the body.
    robot.move_arm(side, 'toward_body')
    # And up to close the circuit.
    robot.move_arm(side, 'up')
