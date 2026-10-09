def flatten_by_pulling_taut(robot, item_name, direction, side, brace_side):
    """Pull a slack flat item taut against its support: the `brace_side` hand
    pins one part of it, the `side` hand drags the other part out, lets the
    fingers walk along, and drags again -- the "Flatten out the <item>"
    subtask for cloth, paper and film.

    The `regrasp` is the maneuver: one drag never flattens a garment, so the
    hand re-bites further along and pulls again. Both hands END CLOSED on the
    item; write their releases where the video shows them leave. Example
    items: 'napkin', 'fabric', 'shirt', 'scarf'; direction e.g. 'right',
    'left' or 'toward_body'.
    """
    # Pin one part of the item.
    robot.grasp(item_name, brace_side)
    # Take the slack part.
    robot.grasp(item_name, side)
    # Drag it out flat.
    robot.move_arm(side, direction)
    # Let the fingers walk further along.
    robot.regrasp(item_name, side)
    # Drag again from the new bite.
    robot.move_arm(side, direction)
