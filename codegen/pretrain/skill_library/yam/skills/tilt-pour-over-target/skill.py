def tilt_pour_over_target(robot, vessel_name, target_name, side, aim_delta, approach_rpy, tilt_deg=90.0):
    """Carry a HELD vessel over a receptacle, tip it out, bring it back upright.

    The "Pour the <stuff> into the <receptacle>" sub-task: locate the
    receptacle, move over it, clear its rim, ONE tip-and-return. The vessel is
    ALREADY HELD when this is called and is STILL HELD, upright, when it
    returns -- putting it back down is the next sub-task.

    Returning to upright is not optional: a vessel left tilted keeps emptying
    into whatever is underneath it.

    Parameters (values below are EXAMPLES, not defaults baked in):
        vessel_name   what is being emptied -- e.g. 'scoop', 'bag', 'cup'
        target_name   what catches it -- e.g. 'container', 'pot', 'bowl'
        side          arm holding the vessel, 'left' or 'right'
        aim_delta     XYZ metres of rim clearance after the coarse move, e.g.
                      [0.0, 0.0, 0.15] -- pour from a height so the stream
                      clears the lip
        approach_rpy  roll-pitch-yaw degrees for the carry pose, e.g.
                      [180.0, 0.0, 0.0] (gripper-down)
        tilt_deg      tip angle; e.g. 90.0 for a scoop or a bag, 180.0 to
                      shake a small container completely empty

    ONE pour per call -- several portions need several calls.
    """
    # find the receptacle in the current frame
    target = robot.detect_objects_oneshot(target_name)
    # carry the vessel over it
    robot.move_arm(side, target.data['detections'][0]['position_3d'], approach_rpy)
    # rise clear of the rim before tipping
    robot.servo_ee_delta(side, aim_delta)
    # tip -- THIS is the pour
    robot.rotate(vessel_name, tilt_deg, side)
    # back upright, or it keeps emptying
    robot.rotate(vessel_name, -tilt_deg, side)
