def stir_contents(robot, tool_name, vessel_name, side):
    """Stir what is in a vessel with the tool ALREADY HELD by the `side` hand:
    bring the tool down inside the vessel and sweep it round.

    One call covers a whole "Stir the <contents> in the <vessel>" subtask.
    Two full sweeps about the vertical axis are written out (skill bodies
    cannot loop); repeat the call for a longer stir.

    Picking the tool up is a separate subtask and is NOT here, and neither is
    the withdrawal: the subtask often ends with the tool still in the vessel.
    If it exits, write that yourself, e.g.
    `goto_relative(side, vessel_name, 'above')` or `move_arm(side, 'up')`.
    """
    # Bring the held tool down inside the vessel, among the contents.
    robot.goto_relative(side, vessel_name, 'inside')
    # Sweep it round the vessel; the grip never changes, nothing is let go.
    robot.rotate(tool_name, 'vertical', 360.0, side)
    robot.rotate(tool_name, 'vertical', 360.0, side)
