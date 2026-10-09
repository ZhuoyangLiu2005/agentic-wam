def tear_off_piece(robot, piece_name, parent_name, tear_side, brace_side, tear_delta):
    """Separate an attached piece from the sheet, roll or bag it is stuck to.

    ONE annotated sub-task slice. The parent is pinned, the free corner of the
    piece is pinched, the piece is pulled along the tear line until the
    adhesive or perforation gives, and it is let go where it lands. Nothing is
    carried anywhere -- fetching the piece and applying it afterwards are
    SEPARATE sub-tasks and belong to the calling program.

    The BRACE is the identity. A sheet, roll or bag that is free to follow the
    tug never separates, so the parent is taken by the other arm before the
    pull. Without it this degenerates into `grasp` + a drag.

    Parameters (values below are EXAMPLES, not baked into the body):
        piece_name   what comes off -- e.g. 'sticker', 'label', 'note', 'tape'
        parent_name  what it comes off -- e.g. 'sticker sheet', 'paper bag',
                     'note pad', 'tape dispenser'
        tear_side    arm that pulls the piece free, e.g. 'right'
        brace_side   arm that pins the parent (must be the OTHER arm)
        tear_delta   XYZ metres of the pull along the tear line, e.g.
                     [0.0, 0.1, 0.05] (label peeled up and away from a sheet)
                     or [0.0, -0.15, 0.1] (sticker off a bag)
    """
    # Pin the parent -- a sheet that travels with the tug never separates.
    robot.grasp(parent_name, brace_side)
    # Pinch the free corner of the piece that is coming off.
    robot.grasp(piece_name, tear_side)
    # Pull along the tear line until the adhesive or perforation gives.
    robot.servo_ee_delta(tear_side, tear_delta)
    # Let the freed piece go where it is -- it is NOT carried in this sub-task.
    robot.open_gripper(tear_side)
