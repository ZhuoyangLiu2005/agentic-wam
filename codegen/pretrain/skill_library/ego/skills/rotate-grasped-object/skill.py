def rotate_grasped_object(robot, object_name, side, axis, angle_deg):
    """Take object_name where it sits with the `side` hand and turn it about a
    semantic axis: the "Rotate the <X>" subtask, spelled with the grasp that
    the `rotate` primitive does NOT include.

    The hand ends STILL ON the object (let go in a later subtask). Covers
    both an object turned in the grip and one spun in place on its support --
    `rotate` is defined over both. Typical values: axis='object_symmetry'
    (a phone, a cap) or 'vertical' (a table, a machine); angle_deg=90.0.
    Example objects: 'phone', 'table', 'dough', 'knob'.
    """
    # Take it where it sits -- no lift.
    robot.grasp(object_name, side)
    # Turn it; the grip does not change.
    robot.rotate(object_name, axis, angle_deg, side)
