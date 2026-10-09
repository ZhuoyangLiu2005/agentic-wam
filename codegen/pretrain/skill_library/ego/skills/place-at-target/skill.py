def place_at_target(robot, object_name, target_name, side, free_side):
    """Set the object carried by the `side` hand down onto/at target_name and
    let the `free_side` hand -- the one that was steadying the object or the
    target -- go as well: the "Place the <X> (in|on|at) <Y>" subtask.

    Pass free_side == side when only one hand was involved (the trailing
    release is then a harmless re-open of an already open hand). Use 'table'
    for target_name when the video shows no finer target. Example pairs:
    ('game piece', 'board'), ('paper cup', 'stack of cups'),
    ('pill', 'pill box'), ('egg', 'egg carton').
    """
    # Set it down at the target and let go with the working hand.
    robot.place(object_name, target_name, side)
    # Free the second hand; the object is down and both hands are clear.
    robot.release(free_side)
