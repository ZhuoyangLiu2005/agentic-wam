def stir_braced_vessel(robot, vessel_name, brace_side, tool_side,
                       plunge_delta, sweep_delta, return_delta):
    """Stir inside a braced vessel with a stirrer the ``tool_side`` arm holds.

    ONE annotated sub-task slice. The vessel is taken, the held stirrer drops
    past the rim into the contents, the brace is RE-SEATED now that the tip is
    loading the wall, and the tip walks one out-and-back sweep. The brace is
    STILL HOLDING when the slice ends.

    What this adds over a run of `servo_ee_delta` is the bimanual brace and its
    re-seat: an unheld cup walks across the board under stirring load, and a
    vessel first taken at the rim rolls in the fingers.

    The stirrer is NOT fetched here -- the spoon or stick must already be in
    the ``tool_side`` gripper and is never let go, which is why no stirrer
    name is in the signature.

    Parameters (values below are EXAMPLES, not baked into the body):
        vessel_name   the braced vessel -- e.g. 'cup', 'bowl', 'pan'
        brace_side    arm that pins the vessel, e.g. 'right'
        tool_side     arm already holding the stirrer (must be the OTHER arm)
        plunge_delta  XYZ metres from over the rim down into the contents,
                      e.g. [0.0, 0.0, -0.15]
        sweep_delta   XYZ metres of one out sweep, e.g. [0.02, 0.02, 0.0]
        return_delta  XYZ metres of the back sweep -- normally the negation of
                      sweep_delta, e.g. [-0.02, -0.02, 0.0]

    ONE sweep pair is written out (skill bodies cannot loop). Call the skill
    again for a longer or thicker mix.
    """
    # Take the vessel -- an unheld cup walks under stirring load.
    robot.grasp(vessel_name, brace_side)
    # Drop the held stirrer past the rim, down into the contents.
    robot.servo_ee_delta(tool_side, plunge_delta)
    # Re-seat the brace now the tip is loading the wall; a rim grip rolls.
    robot.regrasp(vessel_name, brace_side)
    # Circulate the tip: one out-and-back sweep inside the vessel wall.
    robot.servo_ee_delta(tool_side, sweep_delta)
    robot.servo_ee_delta(tool_side, return_delta)
