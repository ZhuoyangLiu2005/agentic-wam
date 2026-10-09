def insert_into_receptacle(robot, object_name, target_name, side, direction):
    """Seat the object ALREADY HELD by the `side` hand into target_name, let
    go, and withdraw the hand along a semantic direction -- the "Insert the
    <X> into the <Y>" subtask.

    `insert` asserts the letting-go, so the hand is EMPTY afterwards. If the
    video shows the hand STILL HOLDING after seating, do not call this: write
    `goto_relative(side, target_name, 'inside')` instead. Example pairs:
    ('battery', 'remote control'), ('cable plug', 'router'),
    ('book', 'book cover'), ('photo', 'envelope'); direction e.g.
    'toward_body'.
    """
    # Seat it into the receptacle and let go.
    robot.insert(object_name, target_name, side)
    # Withdraw the now-empty hand.
    robot.move_arm(side, direction)
