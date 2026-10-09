def pull_open_and_hold(robot, object_name, direction, side):
    """Take hold of a handle/leaf/free end where it sits and draw it open
    along a semantic direction, KEEPING hold of it -- the "Pull open the <X>"
    / "Pull the <X> from the <Y>" subtask for things that stay attached.

    The point is that nothing comes AWAY: a drawer, a stand, a door, a length
    of tape off its roll, a ribbon pulled taut. The hand ENDS CLOSED on it,
    because what comes next varies (a re-bite for more tension, a release, a
    second pull) and is left to the caller. Example objects: 'stand leg',
    'drawer', 'door', 'tape'; direction='right' or 'left'.
    """
    # Take it where it sits.
    robot.grasp(object_name, side)
    # Draw it open / out along the direction.
    robot.move_arm(side, direction)
