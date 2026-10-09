def open_closure(robot, container_name, closure_name, side, brace_side, axis, angle_deg):
    """Open a hinged closure (lid, cover, door, flap, clip, switch) with the
    `side` hand while the `brace_side` hand holds the container body still,
    then come off the closure: the "Open the <X>" subtask.

    The brace is what makes this an open rather than a bare turn -- a friction
    or snap closure lifts the whole container with it unless the body is held.
    The brace OUTLIVES the skill; write its release where that hand leaves.
    Typical values: axis='object_long' (a book cover) or 'lateral' (a lid
    hinged away from the body); angle_deg=+90.0 or +180.0 -- the SIGN is the
    opening sense. Example pairs: ('book', 'book cover'), ('case',
    'case lid'), ('tray', 'tray lid'), ('pill box', 'pill box lid').
    """
    # Hold the container body down so the closure, not the container, moves.
    robot.grasp(container_name, brace_side)
    # Take the closure where it sits.
    robot.grasp(closure_name, side)
    # Swing it open about its hinge.
    robot.rotate(closure_name, axis, angle_deg, side)
    # Come off the closure; it stays open.
    robot.release(side)
