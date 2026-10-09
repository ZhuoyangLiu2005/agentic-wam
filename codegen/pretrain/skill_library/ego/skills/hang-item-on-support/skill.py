def hang_item_on_support(robot, object_name, support_name, side):
    """Bring the item held by the `side` hand OVER a hook/rack/branch and let
    it down onto it -- the "Hang the <X> on the <Y>" subtask.

    The approach from ABOVE is the whole difference from a plain `place`: a
    hang only works if the loop, handle or hook clears the support before the
    hand lets go, so the `goto_relative(..., 'above')` is written as a
    separate line. `place` carries the release, so the hand ENDS EMPTY.
    Example pairs: ('ladle', 'hanging rack'), ('key ring', 'key holder'),
    ('ornament', 'tree'), ('cup', 'rack').
    """
    # Clear the support from above.
    robot.goto_relative(side, support_name, 'above')
    # Let the item down onto it and let go.
    robot.place(object_name, support_name, side)
