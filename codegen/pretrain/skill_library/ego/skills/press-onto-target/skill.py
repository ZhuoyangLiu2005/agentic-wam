def press_onto_target(robot, target_name, side, other_side):
    """Press target_name with both hands and come off it -- the "Press the
    <X> (onto the <Y>)" subtask: seating a lid/case/tag home by force, or
    working a control.

    `press` is a contact-then-off pair, so nothing is held before or after and
    both hands end clear. Pass other_side == side for a one-handed press: the
    second call is then a second dab, which is how a repeated button push is
    written. Example targets: 'lid', 'snap-on cover', 'button', 'keyboard',
    'capsule'.
    """
    # First hand contacts the target and comes off it.
    robot.press(target_name, side)
    # Second hand does the same, seating the other side.
    robot.press(target_name, other_side)
