def fold_edge_over(robot, item_name, left_delta, right_delta):
    """Both arms pinch a cloth edge, carry it across the item, and let go --
    ONE fold.

    The unit of a folding sub-task: grasp the near edge with both arms,
    translate both grippers across the garment (same direction to bring an
    edge over, opposite directions to bring two ends to the middle), then
    release both so the fold lies flat. Longer folds are this body repeated --
    call it again for each further fold.

    Parameters (values below are EXAMPLES, not defaults baked in):
        item_name    the cloth being folded -- e.g. 'towel', 't-shirt',
                     'napkin'
        left_delta   XYZ metres the LEFT gripper carries its corner,
                     e.g. [0.1, 0.0, 0.0] (edge-over) or [0.0, -0.2, 0.0]
        right_delta  XYZ metres for the RIGHT gripper -- the SAME sign as
                     left_delta folds one edge across; the OPPOSITE sign
                     (e.g. [-0.1, 0.0, 0.0]) brings both ends to the middle

    BIMANUAL BY CONSTRUCTION: the arms are the literals 'left' and 'right', so
    the contract is fully concrete.
    """
    # pinch the edge with both arms
    robot.grasp(item_name, 'left')
    robot.grasp(item_name, 'right')
    # carry the pinched edge across the item
    robot.servo_ee_delta('left', left_delta)
    robot.servo_ee_delta('right', right_delta)
    # let the fold lie
    robot.release('left')
    robot.release('right')
