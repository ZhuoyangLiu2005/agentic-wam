def free_hand_and_pick(robot, object_name, pick_side, free_side):
    """Let go with the idle arm, then pick the object up with the working arm.

    A plain "Pick up the <object>" sub-task in which the other arm must let go
    first: the arm that was bracing (or still holding the previous item) OPENS,
    then the working arm picks. Writing them together keeps the open and the
    pick in one sub-task.

    Parameters (values below are EXAMPLES, not defaults baked in):
        object_name  what gets picked up -- e.g. 'pen', 'cup', 'lid'
        pick_side    arm that picks, 'left' or 'right'
        free_side    the OTHER arm, which lets go first -- the brace, or the
                     hand still holding the previous item

    Ends with `pick_side` HOLDING the object clear of the surface and
    `free_side` empty.
    """
    # the idle arm lets go of whatever it was bracing / still holding
    robot.release(free_side)
    # the working arm grasps the object and lifts it clear
    robot.pick(object_name, pick_side)
