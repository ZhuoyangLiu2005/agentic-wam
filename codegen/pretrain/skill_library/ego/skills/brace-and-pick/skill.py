def brace_and_pick(robot, object_name, side, support_name, brace_side):
    """Hold support_name still with the `brace_side` hand and pick object_name
    off/out of it with the `side` hand -- the two-handed "Pick up the <X>"
    subtask, used where the source would otherwise come away with the object.

    The brace OUTLIVES the skill: this body emits no release for `brace_side`;
    write it where the video shows that hand leaving (usually a later
    subtask). Example pairs: ('spoon', 'container'), ('bottle cap',
    'bottle body'), ('pill', 'pill bottle'), ('test tube', 'rack').
    """
    # Hold the source still so the object, not the source, comes away.
    robot.grasp(support_name, brace_side)
    # Lift the object clear; the hand keeps holding it.
    robot.pick(object_name, side)
