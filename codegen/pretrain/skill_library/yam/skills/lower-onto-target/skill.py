def lower_onto_target(robot, target_name, place_side, hover_height, lower_delta):
    """Carry the held object over a DETECTED target and lower it onto/into it.

    The placement half of a pick-and-place: detect the target, read the arm's
    CURRENT wrist orientation so the carried object is not re-oriented on the
    way, move to a hover pose above the detection, then lower with a bounded
    delta.

    Leaves the object at/in the target, STILL HELD. The open is deliberately
    not part of this body -- it often belongs to the next sub-task. Follow with
    `robot.release(place_side)` (or `robot.place` / `robot.attach` /
    `robot.insert`) when the drop belongs to your sub-task.

    Parameters (values below are EXAMPLES, not defaults baked in):
        target_name    where it goes -- e.g. 'container', 'plate', 'box'
        place_side     arm carrying the object, 'left' or 'right'
        hover_height   metres above the detection to arrive at, e.g. 0.1
                       (choose it from the target's height)
        lower_delta    XYZ metres of the descent, e.g. [0.0, 0.0, -0.05]

    GRIPPER-TRANSPARENT: perception + motion only.
    """
    # where is the target, and which way is the wrist pointing right now?
    target = robot.detect_objects_oneshot(target_name)
    state = robot.get_robot_state()
    target_pos = target.data['detections'][0]['position_3d']
    # hover above the target, keeping the CURRENT rpy so the carried object is not re-oriented
    robot.move_arm(place_side, [target_pos[0], target_pos[1], target_pos[2] + hover_height], state.data[place_side]['ee_rpy'])
    # lower it in
    robot.servo_ee_delta(place_side, lower_delta)
