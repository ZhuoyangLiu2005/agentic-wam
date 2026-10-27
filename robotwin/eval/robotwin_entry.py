"""Run RoboTwin's evaluator with the requested episode count.

The pinned upstream script hard-codes 100 episodes in main(). Bind that local
value in memory so its loop and final score use the same requested count.
The external simulator checkout remains unchanged.
"""
import ast
from pathlib import Path
import sys


def _episode_limit(args):
    value = args.get("eval_num_episodes")
    count = 100 if value is None else int(value)
    if count < 1:
        raise ValueError("eval_num_episodes must be positive")
    return count


def _compile_evaluator(source, filename):
    tree = ast.parse(source, filename=filename)
    assignments = []
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name == "main":
            for node in ast.walk(function):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "test_num" for target in node.targets
                ):
                    assignments.append(node)
    if len(assignments) != 1:
        raise RuntimeError("Unsupported RoboTwin evaluator: expected one main() test_num assignment")
    node = assignments[0]
    value = ast.Call(func=ast.Name(id="_agenticwam_episode_limit", ctx=ast.Load()),
                     args=[ast.Name(id="usr_args", ctx=ast.Load())], keywords=[])
    node.value = ast.copy_location(value, node.value)
    return compile(ast.fix_missing_locations(tree), filename, "exec")


def main():
    script = Path("script/eval_policy.py").resolve()
    code = _compile_evaluator(script.read_text(), str(script))
    sys.argv[0] = str(script)
    sys.path.insert(0, str(script.parent))
    namespace = globals()
    namespace.update(__name__="__main__", __file__=str(script),
                     _agenticwam_episode_limit=_episode_limit)
    exec(code, namespace)


if __name__ == "__main__":
    main()
