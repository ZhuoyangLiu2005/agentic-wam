def move_item_two_handed(robot, item_name, side, brace_side, direction, brace_direction):
    """Relocate a bulky item that both hands must hold (coat, box, board,
    shoe, sheet of paper): take it with both hands, carry it along a semantic
    direction, and let go with both -- the "Move the <X> to the designated
    area" subtask.

    Both hands take it WHERE IT SITS (`grasp`, not `pick`): such items are
    slid or dragged rather than lifted clear, and no target object is
    named -- the displacement itself is the meaning. Pass the same value for
    direction and brace_direction for a straight carry; opposite values spread
    the item out. Example items: 'coat', 'shoe', 'shirt', 'box', 'board',
    'sheet of paper'; direction e.g. 'away_from_body'.
    """
    # Both hands take the item where it sits.
    robot.grasp(item_name, brace_side)
    robot.grasp(item_name, side)
    # Carry it.
    robot.move_arm(brace_side, brace_direction)
    robot.move_arm(side, direction)
    # Let it go where it landed.
    robot.release(side)
    robot.release(brace_side)
