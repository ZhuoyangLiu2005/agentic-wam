def pull_part_free(robot, part_name, assembly_name, side, brace_side, direction):
    """Hold an assembly still with the `brace_side` hand and pull part_name
    out of/off it with the `side` hand along a semantic direction -- the
    "Remove the <X> from the <Y>" subtask, for parts that come away by a
    straight pull rather than a lift or a twist.

    BOTH hands end closed: the part is in the `side` hand and the brace is
    still on the assembly. Write both releases where the video shows those
    hands leaving. Example pairs: ('end cap', 'knife handle'), ('key',
    'lock'), ('battery', 'battery cover'), ('plug', 'nozzle');
    direction='right' or 'toward_body'.
    """
    # Hold the assembly so it does not come along.
    robot.grasp(assembly_name, brace_side)
    # Take the part where it sits.
    robot.grasp(part_name, side)
    # Pull it straight free.
    robot.move_arm(side, direction)
