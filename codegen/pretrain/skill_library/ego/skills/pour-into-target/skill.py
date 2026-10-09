def pour_into_target(robot, vessel_name, target_name, side, axis, angle_deg):
    """Pour out of the vessel ALREADY HELD by the `side` hand: bring its mouth
    over target_name, tip it until it empties, and bring it back upright --
    the "Pour <X> into <Y>" subtask.

    Contact-transparent throughout: the grip never changes, so this rides on
    whatever pick/grasp put the vessel in the hand, and the put-down belongs
    to the caller. ONE pour per call -- a scoop-and-pour repetition is written
    as separate subtasks. Example vessels: 'spoon', 'cup', 'beaker', 'test
    tube', 'bottle', 'bag'; axis='object_long' (a spoon) or 'lateral' (a
    bottle), angle_deg e.g. 110.0.
    """
    # Bring the held vessel's mouth over the receiving target.
    robot.goto_relative(side, target_name, 'above')
    # Tip it until it pours.
    robot.rotate(vessel_name, axis, angle_deg, side)
    # Bring it back upright; the pour ends.
    robot.rotate(vessel_name, axis, -angle_deg, side)
