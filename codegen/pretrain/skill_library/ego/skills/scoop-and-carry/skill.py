def scoop_and_carry(robot, source_name, target_name, side):
    """Take a load out of a source container with the utensil ALREADY HELD by
    the `side` hand and bring it over where it is going -- the "Scoop a
    <portion> from the <source>" subtask.

    The whole subtask is three semantic moves: down into the source, straight
    up with the load, across to hang above the target. Contact-transparent:
    the spoon/scoop/spatula is taken in an earlier subtask and is still held
    afterwards, so this skill declares NO gripper events and the pour-out or
    scrape-off is the caller's next line. Example utensils: 'spoon',
    'spatula', 'slotted spoon', 'rice scoop'.
    """
    # Down into the source, under the load.
    robot.goto_relative(side, source_name, 'inside')
    # Lift the loaded utensil clear.
    robot.move_arm(side, 'up')
    # Hold it over where the load is going.
    robot.goto_relative(side, target_name, 'above')
