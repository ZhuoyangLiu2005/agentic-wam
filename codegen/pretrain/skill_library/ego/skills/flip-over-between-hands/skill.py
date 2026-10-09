def flip_over_between_hands(robot, object_name, axis, angle_deg, side, other_side):
    """Turn an object face-over by passing it between the hands: the `side`
    hand takes it, the `other_side` hand lets go so it can swing, the `side`
    hand turns it through a half-turn, and the `other_side` hand takes the
    new face -- the "Flip the <X>" subtask.

    The hand swap is what makes this a flip rather than a `rotate`: a two-hand
    hold cannot carry an object through 180 degrees, so one hand must let go
    for the turn and re-take afterwards. Both hands END CLOSED on the object.
    Example objects: 'pencil', 'book', 'box lid', 'remote control';
    axis='lateral', 'object_long' or 'vertical', angle_deg typically 180.0.
    """
    # The turning hand takes it.
    robot.grasp(object_name, side)
    # The other hand lets go so it can swing.
    robot.release(other_side)
    # Half-turn it face-over.
    robot.rotate(object_name, axis, angle_deg, side)
    # The other hand takes the new face.
    robot.grasp(object_name, other_side)
