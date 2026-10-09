def relocate_object(robot, object_name, target_name, side):
    """Move an object from where it sits to a named target: lift it clear and
    set it down there.

    The "Move the <object> to the <target>" sub-task -- ONE arm, pick then
    place, both semantic verbs, in a single slice. The object leaves the
    surface (that is what separates a move from an adjust) and the gripper
    ends open.

    Parameters (values below are EXAMPLES, not defaults baked in):
        object_name  what is moved -- e.g. 'container', 'box', 'book', 'vase'
        target_name  where it goes -- e.g. 'table', 'bin', 'tray'
        side         arm that carries it, 'left' or 'right'

    Ends with the gripper OPEN and the object resting at the target.
    """
    # lift it clear of the surface
    robot.pick(object_name, side)
    # set it down at the target and let go
    robot.place(object_name, target_name, side)
