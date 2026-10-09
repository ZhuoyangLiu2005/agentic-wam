def fold_over_and_crease(robot, item_name, side, brace_side, direction, fold_direction):
    """Fold a flat item (cloth, paper, bag, sleeve) over onto itself: brace it
    with the `brace_side` hand, take the free edge with the `side` hand, carry
    that edge across in two semantic steps, let it down, and press the crease.

    ONE fold per call -- a multi-fold garment is one subtask per fold, so
    call this once per annotated fold. The brace OUTLIVES the skill; write
    its release where that hand leaves. Example items: 'paper', 'shirt',
    'napkin', 'wrapping paper', 'sleeve'; direction='toward_body' then
    fold_direction='left' gives a diagonal fold.
    """
    # Hold the item down so the fold, not the whole item, travels.
    robot.grasp(item_name, brace_side)
    # Take the free edge with the working hand.
    robot.grasp(item_name, side)
    # Carry the edge across the item ...
    robot.move_arm(side, direction)
    # ... and land it on the far side.
    robot.move_arm(side, fold_direction)
    # Let the folded edge down.
    robot.release(side)
    # Press the crease flat.
    robot.press(item_name, side)
