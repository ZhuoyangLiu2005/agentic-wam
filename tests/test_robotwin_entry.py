"""Upstream's fixed 100 must not turn a smoke job into a full benchmark."""
import pytest
import subprocess
import sys

from robotwin.eval import robotwin_entry
from robotwin.eval.robotwin_entry import _compile_evaluator, _episode_limit


@pytest.mark.parametrize("count", [1, 3, None])
@pytest.mark.parametrize("upstream_assignment", ["100", "usr_args.get('eval_num_episodes', 100)"])
def test_requested_count_controls_loop_and_score(count, upstream_assignment):
    source = f"""
def main(usr_args):
    test_num = {upstream_assignment}
    episodes = list(range(test_num))
    return episodes, len(episodes) / test_num
"""
    namespace = {"_agenticwam_episode_limit": _episode_limit}
    exec(_compile_evaluator(source, "fixture_eval.py"), namespace)
    episodes, score = namespace["main"]({"eval_num_episodes": count})
    assert len(episodes) == (100 if count is None else count)
    assert score == 1.0


@pytest.mark.parametrize("count", [0, -1])
def test_invalid_episode_count_fails_before_evaluation(count):
    with pytest.raises(ValueError, match="must be positive"):
        _episode_limit({"eval_num_episodes": count})


def test_unknown_upstream_layout_fails_instead_of_running_unbounded():
    with pytest.raises(RuntimeError, match="Unsupported RoboTwin evaluator"):
        _compile_evaluator("def main(usr_args): pass", "fixture_eval.py")


def test_entrypoint_preserves_main_module_and_external_source(tmp_path):
    script = tmp_path / "script/eval_policy.py"
    script.parent.mkdir()
    source = """
import sys
def main(usr_args):
    test_num = 100
    assert sys.modules['__main__'].main is main
    print(test_num)
if __name__ == '__main__':
    main({'eval_num_episodes': 1})
"""
    script.write_text(source)
    result = subprocess.run([sys.executable, robotwin_entry.__file__], cwd=tmp_path,
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "1"
    assert script.read_text() == source
