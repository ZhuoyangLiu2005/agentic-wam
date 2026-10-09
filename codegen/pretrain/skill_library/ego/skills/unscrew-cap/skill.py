def unscrew_cap(robot, cap_name, side, axis, angle_deg):
    """Unscrew a threaded part (cap, lid, nipple, pump head, bulb, jar) with
    the `side` hand: take it, twist, let the wrist unwind and re-close, twist
    again, and lift the freed part clear -- the "Unscrew the <X> from the <Y>"
    subtask.

    TWO twist cycles are unrolled because skill bodies are straight-line; a
    tighter thread is a literal `for` loop over close_hand/rotate/open_hand in
    the program, not this skill. The wrist reset is written as `regrasp` --
    the ego API's own name for the open-then-re-close pair. The hand ENDS
    HOLDING the freed part. Example parts: 'bottle cap', 'lid', 'pump head',
    'light bulb', 'jar'; axis='object_symmetry' or 'vertical', angle_deg=90.0.
    """
    # Take the threaded part where it sits.
    robot.grasp(cap_name, side)
    # First twist.
    robot.rotate(cap_name, axis, angle_deg, side)
    # Let the wrist unwind and re-close on the part.
    robot.regrasp(cap_name, side)
    # Second twist; the thread runs free.
    robot.rotate(cap_name, axis, angle_deg, side)
    # Lift the freed part clear of its body.
    robot.pick(cap_name, side)
