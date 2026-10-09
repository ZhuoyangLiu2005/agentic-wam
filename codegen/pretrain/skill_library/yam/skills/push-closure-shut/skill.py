def push_closure_shut(robot, closure_name, side, close_delta):
    """Grip a drawer / lid / door / cover and drive it through its travel until
    it is shut, then let go.

    The "Close the <thing>" sub-task. Unlike an adjust, the thing being moved is
    HINGED OR SLIDING: it has one degree of freedom and the delta runs along it
    until the closure seats. The gripper opens at the end because the closure
    stays where it was driven.

    Parameters (values below are EXAMPLES, not defaults baked in):
        closure_name  what is driven shut -- e.g. 'drawer', 'lid', 'door',
                      'box cover'
        side          arm that pushes, 'left' or 'right'
        close_delta   XYZ metres along the closure travel, e.g.
                      [-0.3, 0.0, 0.1] (swing a cover over),
                      [0.2, 0.0, 0.0] (push a door shut),
                      [0.15, 0.0, 0.0] (push a cap on)

    Ends with the gripper OPEN and the closure shut.
    """
    # take the drawer front / lid edge / cover
    robot.grasp(closure_name, side)
    # drive it along its travel until it seats
    robot.servo_ee_delta(side, close_delta)
    # let go -- it stays shut
    robot.release(side)
