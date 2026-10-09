def flip_object_over(robot, object_name, side, lift_delta, turn_deg=180.0):
    """Lift a held flat object, roll the wrist over, drop it face-down, re-pin it.

    The "Turn over the <object>" sub-task: the object is ALREADY HELD; the arm
    lifts it just clear of the surface, rolls the wrist a half turn, and
    RELEASES -- gravity finishes the flip -- then grasps again to pin the
    now-inverted object down for whatever the next sub-task does to it.

    Parameters (values below are EXAMPLES, not defaults baked in):
        object_name  the flat thing -- e.g. 'glove', 'napkin', 'file folder'
        side         arm doing the flip, 'left' or 'right'
        lift_delta   XYZ metres of clearance before the roll, e.g.
                     [0.0, 0.0, 0.1]
        turn_deg     wrist roll; 180.0 is the default (a full flip),
                     90.0 to stand something on edge

    NOT for turning a screw cap (``ratchet_twist_off``) and not for turning a
    page (a bare ``servo_ee_delta`` sweep with whatever is in hand).
    """
    # lift just clear so the object can swing
    robot.servo_ee_delta(side, lift_delta)
    # roll the wrist -- this is the flip
    robot.rotate(object_name, turn_deg, side)
    # let it fall the last centimetre, landing inverted
    robot.release(side)
    # pin the flipped object down again for the next sub-task
    robot.grasp(object_name, side)
