def wipe_surface_back_and_forth(robot, surface_name, relation, stroke_direction, return_direction, side):
    """Work a cloth/sponge/brush ALREADY HELD by the `side` hand over a
    surface: park the hand on the surface, then two out-and-back strokes --
    the "Wipe the <surface>" subtask.

    Contact-transparent from end to end: the cloth is picked up in an earlier
    subtask and is still in the hand afterwards, so this skill declares NO
    gripper events. TWO stroke pairs are unrolled because skill bodies are
    straight-line; wrap the call in a literal `for` loop for a longer scrub.
    Typical values: relation='above' or 'against'; stroke/return =
    ('left','right') or ('toward_body','away_from_body').
    """
    # Park the cloth on the surface.
    robot.goto_relative(side, surface_name, relation)
    # Out and back.
    robot.move_arm(side, stroke_direction)
    robot.move_arm(side, return_direction)
    # And again.
    robot.move_arm(side, stroke_direction)
    robot.move_arm(side, return_direction)
