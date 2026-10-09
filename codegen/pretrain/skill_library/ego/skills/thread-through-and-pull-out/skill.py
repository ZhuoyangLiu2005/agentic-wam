def thread_through_and_pull_out(robot, object_name, target_name, pull_direction, side, other_side):
    """Push a lace/cord/pin through a hole in target_name with the `side`
    hand, then take the emerging end with the `other_side` hand and draw it
    through -- the "Pass the <cord> through the <X>" subtask.

    The hand swap is the whole maneuver: `insert` lets go of the leading end
    on the near side, the same hand then pins the target so it cannot travel,
    and the far hand -- which must let go of whatever it held -- catches the
    end that came out and pulls. Both hands end CLOSED: `side` on the target,
    `other_side` on the drawn cord. Example pairs: ('shoelace','shoe'),
    ('pin','fabric'); pull_direction='right'.
    """
    # Push the leading end through the hole and let it go.
    robot.insert(object_name, target_name, side)
    # That hand now pins the target instead.
    robot.grasp(target_name, side)
    # Free the far hand.
    robot.release(other_side)
    # Catch the end that came out.
    robot.grasp(object_name, other_side)
    # Draw the slack through.
    robot.move_arm(other_side, pull_direction)
