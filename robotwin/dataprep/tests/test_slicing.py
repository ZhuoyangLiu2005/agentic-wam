"""Contract + engine tests on a synthetic program (no data needed).
Run: python robotwin/dataprep/tests/test_slicing.py   (or python -m pytest <this dir> -q)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hybrid_eval import assemble  # noqa: E402
from dataprep import slicing  # noqa: E402

PROGRAM = '''\
import numpy as np

LIFT = 0.1

def reach(hand, pos) -> "True when the hand got there":
    return goto_pose(hand, pos)

def audit(x):
    print("AUDIT", x)

# [setup]
(pos, _) = get_object_pose("cup")
hand = "left" if pos[0] < 0 else "right"
print("PERCEIVE", pos)
# [stage 1/3] grasp the cup
for attempt in (0.05, 0.0):
    if reach(hand, pos):
        break
close_gripper(hand)
# [setup]
up = pos + np.array([0, 0, LIFT])
# [stage 2/3] lift it
if up[2] > 1.0:
    raise RuntimeError("too high")
goto_pose(hand, up)
# [stage 3/3] set it down and let go
reach(hand, pos)
open_gripper(hand)
# [omit]
audit(pos)
'''


def _prog(tmp_path, text=PROGRAM, executed=None):
    p = tmp_path / "frozen_program.py"
    p.write_text(text)
    return slicing.Program(p, executed_src=executed)


def test_sections_and_texts(tmp_path):
    prog = _prog(tmp_path)
    assert prog.check() == []
    assert prog.n_stages == 3
    pre = prog.preamble()
    assert 'def reach(hand, pos) -> "True when the hand got there": ...' in pre
    assert "def audit(x): ..." in pre
    assert "# Setup:\n(pos, _) = get_object_pose('cup')" in pre
    assert "up = pos + np.array([0, 0, LIFT])" in pre            # reopened setup section
    assert "print(" not in pre                                     # prints stripped
    s = prog.slices()
    assert s[0].startswith("# [stage 1/3] grasp the cup\n")
    assert "raise" not in s[1] and "goto_pose(hand, up)" in s[1]  # `if: raise` guard stripped
    assert "audit" not in "".join(s)                               # omitted statement
    for k, text in enumerate(s, 1):
        assert assemble.parse_header(text) == (k, 3, "header")


def test_contract_errors(tmp_path):
    bad = PROGRAM.replace("# [stage 2/3] lift it", "# [stage 3/3] lift it")
    errs = _prog(tmp_path, bad).check()
    assert any("contiguous" in e for e in errs)
    bad = PROGRAM.replace("# [omit]\naudit(pos)", "# [omit]\nopen_gripper(hand)")
    assert any("omitted statements issue motions" in e for e in _prog(tmp_path, bad).check())


def test_equivalence_with_executed_program(tmp_path):
    executed = "\n".join(l for l in PROGRAM.splitlines() if not l.startswith("# [")) + "\n"
    executed = executed.replace(' -> "True when the hand got there"', "")
    prog = _prog(tmp_path, executed=executed)                     # comment/annotation-only diff: accepted
    assert prog.exec_main is not None
    try:
        _prog(tmp_path, executed=executed.replace("LIFT = 0.1", "LIFT = 0.2"))
        assert False, "a code change must be rejected"
    except ValueError:
        pass


def test_frames_with_top_line(tmp_path):
    prog = _prog(tmp_path)
    L = {l.strip(): i for i, l in enumerate(PROGRAM.splitlines(), 1)}
    reach_line = L['return goto_pose(hand, pos)']
    recs = [  # stage 1: reach (via helper) + close; stage 2: goto; stage 3: reach (via helper) + open
        {"line": reach_line, "top_line": L["for attempt in (0.05, 0.0):"], "api": "goto_pose", "frame_start": 0, "frame_end": 20},
        {"line": L["close_gripper(hand)"], "top_line": L["close_gripper(hand)"], "api": "close_gripper", "frame_start": 20, "frame_end": 30},
        {"line": L["goto_pose(hand, up)"], "top_line": L["goto_pose(hand, up)"], "api": "goto_pose", "frame_start": 30, "frame_end": 45},
        {"line": reach_line, "top_line": L["reach(hand, pos)"], "api": "goto_pose", "frame_start": 45, "frame_end": 60},
        {"line": L["open_gripper(hand)"], "top_line": L["open_gripper(hand)"], "api": "open_gripper", "frame_start": 60, "frame_end": 70},
    ]
    assert prog.stage_of_records(recs) == [1, 1, 2, 3, 3]
    assert prog.stage_frames(recs, 70) == [(0, 30), (30, 45), (45, 70)]
    # a stage that never issued a motion is zero-length at the next stage's start
    assert prog.stage_frames([recs[0], recs[1], recs[3], recs[4]], 70) == [(0, 45), (45, 45), (45, 70)]
    # without top_line the shared helper is ambiguous: the heuristic keeps the current stage and warns
    for r in recs:
        r.pop("top_line")
    prog.stage_of_records(recs)
    assert prog.ambiguous > 0


def test_build_layout(tmp_path):
    try:
        import torch  # noqa: F401  (build writes cycle_by_idx.pt; the training loader reads it with torch)
    except ImportError:
        print("skip test_build_layout (no torch)"); return
    prog = _prog(tmp_path)
    L = {l.strip(): i for i, l in enumerate(PROGRAM.splitlines(), 1)}
    eps = [dict(seed=1, num_frames=50, records=[
        {"line": L["close_gripper(hand)"], "top_line": L["close_gripper(hand)"], "api": "close_gripper", "frame_start": 0, "frame_end": 10},
        {"line": L["goto_pose(hand, up)"], "top_line": L["goto_pose(hand, up)"], "api": "goto_pose", "frame_start": 10, "frame_end": 30},
        {"line": L["open_gripper(hand)"], "top_line": L["open_gripper(hand)"], "api": "open_gripper", "frame_start": 30, "frame_end": 50}])]
    out = tmp_path / "cache"
    rows = slicing.build(prog, eps, out, n_train=1)
    assert rows[0]["phases"] == [(0, 10), (10, 30), (30, 50)]
    assert (out / "preamble.txt").read_text() == prog.prompt_preamble()   # prompt form: stubs of stage-called helpers + setup names
    assert not (out / "setup.py").exists()                               # the eval derives the setup from the frozen program
    assert "get_object_pose" in prog.exec_setup() and "# Setup" not in prog.exec_setup()
    assert "# Setup:" not in (out / "preamble.txt").read_text() and " -> " not in (out / "preamble.txt").read_text()
    assert prog.preamble() == prog.prompt_preamble(style="full")
    assert json.load(open(out / "episodes_train.json")) == [0] and json.load(open(out / "episodes_val.json")) == []
    ep = json.load(open(out / "slices_ep" / "ep_000000.json"))
    assert ep["slices"] == prog.slices()
    import torch
    cyc = torch.load(out / "cycle_by_idx.pt")["cycle_by_idx"]
    assert cyc.tolist() == [0] * 10 + [1] * 20 + [2] * 20


def test_unparse_is_interpreter_independent(tmp_path):
    prog = _prog(tmp_path)
    assert "(pos, _) = get_object_pose('cup')" in prog.preamble()      # 3.10 rendering on every Python


LOOP_PROGRAM = '''\
import numpy as np

NAMES = ("large", "medium", "small")
PLACE_ORDER = (2, 1, 0)

def move_block(name, slot) -> "hand that moved it":
    goto_pose("left", slot)
    close_gripper("left")
    goto_pose("left", slot)
    open_gripper("left")
    return "left"

# [setup]
slots = perceive_slots()
last = None
# [stage each] pick and place the {NAMES[idx]} block
for idx in PLACE_ORDER:
    name = NAMES[idx]
    last = move_block(name, slots[name])
# [stage 2/2] park the arm
goto_pose("left", HOME)
'''


def _loop_records(L):
    """Two executed iterations (idx 2 then 1) + the park; motions come from the helper, top_line = the loop."""
    loop_line, helper_line, park = L["for idx in PLACE_ORDER:"], L['goto_pose("left", slot)'], L['goto_pose("left", HOME)']
    recs, f = [], 0
    for idx in (2, 1):
        for api in ("goto_pose", "close_gripper", "goto_pose", "open_gripper"):
            recs.append({"line": helper_line, "top_line": loop_line, "top_locals": {"idx": idx, "name": "x"},
                         "api": api, "frame_start": f, "frame_end": f + 10}); f += 10
    recs.append({"line": park, "top_line": park, "top_locals": {}, "api": "goto_pose", "frame_start": f, "frame_end": f + 5})
    return recs, f + 5


def test_loop_stage_contract(tmp_path):
    prog = _prog(tmp_path, LOOP_PROGRAM)
    assert prog.check() == [] and prog.n_stages == 2 and prog.each == {1: "pick and place the {NAMES[idx]} block"}
    L = {l.strip(): i for i, l in enumerate(LOOP_PROGRAM.splitlines(), 1)}
    recs, n = _loop_records(L)
    s = prog.slices(records=recs)
    assert [x.splitlines()[0] for x in s] == ["# [stage 1/3] pick and place the small block",
                                              "# [stage 2/3] pick and place the medium block",
                                              "# [stage 3/3] park the arm"]
    assert "name = NAMES[2]" in s[0] and "name = NAMES[1]" in s[1]      # loop target bound per iteration
    assert prog.stage_of_records(recs) == [1] * 4 + [2] * 4 + [3]
    assert prog.stage_frames(recs, n) == [(0, 40), (40, 80), (80, 85)]
    # an episode that ran only one iteration has one stage fewer (park then starts at frame 40)
    one = recs[:4] + [dict(recs[-1], frame_start=40, frame_end=45)]
    assert len(prog.slices(records=one)) == 2 and prog.stage_frames(one, 45) == [(0, 40), (40, 45)]


def test_loop_stage_errors(tmp_path):
    bad = LOOP_PROGRAM.replace("# [stage each] pick and place the {NAMES[idx]} block\nfor idx in PLACE_ORDER:",
                               "# [stage each] pick\nx = 1\nfor idx in PLACE_ORDER:")
    assert any("exactly one top-level `for`" in e for e in _prog(tmp_path, bad).check())


if __name__ == "__main__":                                              # plain runner, no pytest required
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print("ok", name)
