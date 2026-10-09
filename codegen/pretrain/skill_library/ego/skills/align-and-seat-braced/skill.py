def align_and_seat_braced(robot, object_name, target_name, side, brace_side):
    """Pin the target with the `brace_side` hand and push the part home into
    it with the `side` hand -- the "Align and insert/fit the <part> into the
    <target>" subtask, done as a two-hand split.

    The brace is what makes the alignment possible: a puzzle base, a panel, a
    fabric corner would slide away under the seating force. `insert` asserts
    the letting-go, so the `side` hand ENDS EMPTY while the brace hand is
    STILL ON the target -- write its release where the video shows it leave.
    Where the part ends FIXED by contact force rather than seated in a socket,
    write `attach` at the call site instead of calling this skill. Example
    pairs: ('puzzle piece', 'base board'), ('shelf panel', 'back panel'),
    ('clip', 'pin').
    """
    # Pin the target so the alignment survives the push.
    robot.grasp(target_name, brace_side)
    # Push the part home into it and let go.
    robot.insert(object_name, target_name, side)
