def approach_best_grasp(robot, object_name, approach_side, close_in_delta):
    """Sample grasps for an object, take the best reachable one, and bring the
    gripper onto it -- the standard three-line approach plus the final nudge.

    A common grasp-approach motif: sample candidates, rank them for THIS arm,
    drive the end-effector to the winner, then close the last centimetres
    with a bounded delta. It leaves the gripper as it found it, parked on the
    grasp pose; the caller decides what happens next (`robot.close_gripper`,
    `robot.grasp`, a rotate, a press).

    Parameters (values below are EXAMPLES, not defaults baked in):
        object_name      what to approach -- e.g. 'knife', 'fork', 'jar lid'
        approach_side    arm that approaches, 'left' or 'right'
        close_in_delta   XYZ metres of the last bounded nudge onto the object,
                         e.g. [0.0, 0.0, -0.05] to settle down onto it

    GRIPPER-TRANSPARENT on purpose: nothing here opens or closes, so it can be
    prefixed to any grasping sub-task without disturbing that sub-task's
    contract.
    """
    # rank the sampled grasps by reachability for THIS arm
    candidates = robot.sample_grasp_pose(object_name)
    best = robot.select_best_grasp(candidates.data['grasps'], side=approach_side)
    # drive the end-effector onto the winning grasp pose
    robot.move_arm(approach_side, best.data['grasp'].position, best.data['grasp'].rpy)
    # close the last centimetres with a bounded nudge
    robot.servo_ee_delta(approach_side, close_in_delta)
