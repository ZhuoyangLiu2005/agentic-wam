def scrub_back_and_forth(robot, surface_name, wipe_side, stroke_delta, return_delta):
    """Bring the held cloth to a detected surface and scrub it once, back and
    forth.

    The "Wipe the <surface>" sub-task at slice altitude: detect the thing being
    cleaned, read the CURRENT wrist orientation so the cloth stays flat against
    it, drive there, then two opposed bounded strokes. The cloth is already in
    the gripper -- picking it up is a different sub-task.

    Parameters (values below are EXAMPLES, not defaults baked in):
        surface_name   what is wiped -- e.g. 'screen', 'plate', 'table', 'box'
        wipe_side      arm holding the cloth, 'left' or 'right'
        stroke_delta   XYZ metres of the forward stroke, e.g. [0.05, 0.0, 0.0]
                       or [0.0, 0.05, 0.0]
        return_delta   XYZ metres of the back stroke -- normally the NEGATION
                       of stroke_delta, e.g. [-0.05, 0.0, 0.0]

    GRIPPER-TRANSPARENT: the cloth is never let go, so there is no gripper
    contract. Call again for more strokes.
    """
    # where is the surface, and how is the wrist oriented right now?
    surface = robot.detect_objects_oneshot(surface_name)
    state = robot.get_robot_state()
    # bring the cloth onto the surface, keeping the current rpy so it stays flat
    robot.move_arm(wipe_side, surface.data['detections'][0]['position_3d'], state.data[wipe_side]['ee_rpy'])
    # one back-and-forth scrub
    robot.servo_ee_delta(wipe_side, stroke_delta)
    robot.servo_ee_delta(wipe_side, return_delta)
