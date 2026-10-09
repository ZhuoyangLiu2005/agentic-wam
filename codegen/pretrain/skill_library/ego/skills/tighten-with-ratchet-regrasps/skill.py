def tighten_with_ratchet_regrasps(robot, object_name, relation, axis, angle_deg, side):
    """Reach onto an ALREADY-SEATED cap/fitting and take up the last of its
    thread with the `side` hand: turn, let the bite go, re-bite, turn again --
    the "Tighten the <X> on/onto the <Y>" subtask.

    Distinguished from screwing a part on from loose by the phase: the hand
    ARRIVES already on the part and turns FIRST (the thread is engaged and the
    part no longer moves freely), and the subtask ends with the fingers open
    on it. The off hand normally braces the body being tightened against --
    write that `grasp` at the call site, it is not in this body. Example
    objects: 'lid', 'bottle cap', 'fitting'; relation='above',
    axis='object_symmetry', angle_deg=90.0.
    """
    # Bring the hand onto the seated part.
    robot.goto_relative(side, object_name, relation)
    # Take up thread as far as the wrist allows.
    robot.rotate(object_name, axis, angle_deg, side)
    # Let the bite go and take a fresh one.
    robot.open_hand(side)
    robot.close_hand(side)
    # Second pull on the thread.
    robot.rotate(object_name, axis, angle_deg, side)
    robot.open_hand(side)
