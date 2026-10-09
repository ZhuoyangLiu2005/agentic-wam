def close_closure(robot, closure_name, side, other_side, axis, angle_deg):
    """Swing a closure shut with BOTH hands on it and come off it: the "Close
    the <X>" subtask for lids, covers, cases and doors closed two-handed.

    Both hands take the closure, both swing it, both let go; nothing is held
    afterwards. angle_deg is the CLOSING sense -- the negation of what
    `open_closure` uses on the same closure. Example closures: 'case lid',
    'tray lid', 'machine lid', 'pot lid', 'storage box lid', 'backpack';
    e.g. axis='lateral', angle_deg=-90.0.
    """
    # Both hands take the closure.
    robot.grasp(closure_name, other_side)
    robot.grasp(closure_name, side)
    # Both swing it shut.
    robot.rotate(closure_name, axis, angle_deg, other_side)
    robot.rotate(closure_name, axis, angle_deg, side)
    # Both come off it; it stays shut.
    robot.release(other_side)
    robot.release(side)
