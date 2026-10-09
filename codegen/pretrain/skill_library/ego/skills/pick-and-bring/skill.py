def pick_and_bring(robot, object_name, side, target_name, relation):
    """Pick object_name up with the `side` hand and carry it to a semantic
    relation on target_name -- the "Pick up the <X>" subtask written as
    the lift AND the carry that follows it, in one slice.

    The hand ENDS HOLDING the object; the put-down (place/insert/attach) is
    the next subtask and belongs to the caller. Example objects: 'game
    piece', 'paper cup', 'pill box', 'test tube', 'egg', 'lid'; the
    relation is usually 'above' (the object is brought over where it is
    going next), with 'in_front_of' and 'inside' also possible.
    """
    # Take it off its support; the hand keeps holding it.
    robot.pick(object_name, side)
    # Carry it to where it is going next; the grip does not change.
    robot.goto_relative(side, target_name, relation)
