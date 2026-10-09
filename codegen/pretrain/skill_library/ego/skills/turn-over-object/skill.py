def turn_over_object(robot, object_name, side, axis, angle_deg, target_name, relation):
    """Flip the object held by the `side` hand face-over and bring the new
    face to a semantic relation on target_name -- the "Turn over the <X>"
    subtask.

    Contact-transparent: the object must ALREADY be in the `side` hand and is
    still there afterwards. Distinguished from `rotate_grasped_object` by the
    half-turn and the presentation that follows it -- a turn-over is always
    "so that the other face is now facing the work". Example objects: 'box',
    'shoe', 'bottle cap', 'card', 'sock'; axis='lateral', angle_deg=180.0,
    relation='above'.
    """
    # Flip it over.
    robot.rotate(object_name, axis, angle_deg, side)
    # Present the new face where it is going.
    robot.goto_relative(side, target_name, relation)
