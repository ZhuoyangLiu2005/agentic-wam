def hand_over_hand_extract(robot, object_name, side, pull_delta):
    """Two grip-pull-release bites that walk an object out of a bag/slot/sleeve.

    The "Take out the <object> from the <container>" sub-task. A long object
    does not clear its container in one pull -- the arm runs out of workspace,
    so it lets go, takes a fresh bite further along, and pulls again. Two
    bites are written out (skill bodies cannot loop); call it again if the
    object is still captive.

    Parameters (values below are EXAMPLES, not defaults baked in):
        object_name  what is coming out -- e.g. 'glove', 'racket handle',
                     'note'
        side         arm doing the extraction, 'left' or 'right'
        pull_delta   XYZ metres of one extraction pull, along the mouth of the
                     container, e.g. [0.0, -0.1, 0.0] or [0.15, -0.15, 0.0]

    ENDS RELEASED -- the object is out and free; picking it up to carry it
    somewhere is the next sub-task. The container is usually braced by the
    other arm (also its own sub-task).
    """
    # bite 1 -- grip where it sits, drag it partway out, let go
    robot.grasp(object_name, side)
    robot.servo_ee_delta(side, pull_delta)
    robot.release(side)
    # bite 2 -- fresh grip further down, drag it the rest of the way
    robot.grasp(object_name, side)
    robot.servo_ee_delta(side, pull_delta)
    robot.release(side)
