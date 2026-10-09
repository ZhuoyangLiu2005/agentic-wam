def wrap_tension_pull(robot, strand_name, side, pull_delta):
    """Two hand-over-hand bites on a strand being wound: re-seat, pull taut, again.

    The "Wrap the <strand> around the <core>" sub-task. A lace, cable or tape
    band cannot be wound in one continuous motion with a two-finger gripper, so
    the arm alternates ``regrasp`` (a fresh bite further along) with a short
    pull that takes up the slack. Two bites is the straight-line unroll of that
    cycle; call it again for another turn.

    Parameters (values below are EXAMPLES, not defaults baked in):
        strand_name  the thing being wound -- e.g. 'shoelace', 'cable', 'tape',
                     'elastic band'
        side         arm taking the bites, 'left' or 'right'
        pull_delta   XYZ metres of one tensioning pull, e.g.
                     [0.0, 0.1, 0.05] on a shoelace, [-0.15, 0.0, 0.0] on a
                     cable being drawn onto a reel

    Gripper-transparent at the ends: ``regrasp`` opens and closes internally,
    so the strand is HELD before and after.
    """
    # bite 1 -- fresh grip further along the strand, then take up the slack
    robot.regrasp(strand_name, side)
    robot.servo_ee_delta(side, pull_delta)
    # bite 2 -- same again; the wrap tightens one turn per call
    robot.regrasp(strand_name, side)
    robot.servo_ee_delta(side, pull_delta)
