"""Map an obs-grounded skill (Step-3 labeler) to a function-like `policy_chunk` for the CURRENT
sub-goal. The code stays structured (real reasoning space) but now varies per chunk with the live
state, instead of being one constant code per instruction. Same skill/robot API as coding/prompt.py.
"""
from __future__ import annotations

from agenticwam.coding.skill_library import ID_TO_NAME
from agenticwam.coding.policy_code_templates import _detect_objects, _q


def skill_to_policy_code(skill_id: int, instruction: str) -> str:
    name = ID_TO_NAME.get(int(skill_id), "DONE")
    objs = _detect_objects(instruction or "")
    obj = objs[0] if objs else "object"
    target = objs[1] if len(objs) > 1 else (objs[0] if objs else "object")
    L = ["def policy_chunk(obs, state, robot, skills):"]
    L.append(f"    {obj} = skills.detect({_q(obj)}, obs)")
    if name == "GRASP":
        L += [f"    if not skills.check_attached({obj}, robot.right_arm.gripper):",
              f"        g = skills.sample_grasp({obj})",
              "        robot.right_arm.move_to(g.pregrasp)",
              "        robot.right_arm.move_to(g.grasp)",
              "        robot.right_arm.close_gripper()",
              f"    return skills.check_attached({obj}, robot.right_arm.gripper)"]
    elif name == "LIFT":
        L += ["    robot.right_arm.delta_pose(0.0, 0.0, 0.15, 0.0, 0.0, 0.0)",
              f"    return skills.check_attached({obj}, robot.right_arm.gripper)"]
    elif name in ("PLACE", "ALIGN"):
        L += [f"    {target} = skills.detect({_q(target)}, obs)",
              f"    robot.right_arm.move_to(skills.sample_place_pose({obj}, {target}))",
              "    robot.right_arm.open_gripper()",
              "    return skills.visual_diff(obs, obs)"]
    elif name == "INSERT":
        L += [f"    {target} = skills.detect({_q(target)}, obs)",
              f"    robot.right_arm.move_to(skills.place_inside({obj}, {target}))",
              "    robot.right_arm.open_gripper()",
              f"    return skills.verify_inside({obj}, {target})"]
    elif name == "PRESS":
        L += [f"    robot.right_arm.move_to(skills.sample_place_pose({obj}, {obj}))",
              "    robot.right_arm.delta_pose(0.0, 0.0, -0.05, 0.0, 0.0, 0.0)",
              "    return skills.visual_diff(obs, obs)"]
    elif name == "PUSH":
        L += [f"    robot.right_arm.move_to(skills.sample_place_pose({obj}, {obj}))",
              "    return skills.visual_diff(obs, obs)"]
    elif name in ("OPEN", "CLOSE"):
        sign = "-0.2" if name == "OPEN" else "0.2"
        L += [f"    h = skills.sample_grasp({obj})",
              "    robot.right_arm.move_to(h.grasp)",
              "    robot.right_arm.close_gripper()",
              f"    robot.right_arm.delta_pose({sign}, 0.0, 0.0, 0.0, 0.0, 0.0)",
              "    robot.right_arm.open_gripper()",
              "    return skills.visual_diff(obs, obs)"]
    elif name == "HANDOVER":
        L += [f"    g = skills.sample_grasp({obj})",
              "    robot.left_arm.move_to(g.grasp)",
              "    robot.left_arm.close_gripper()",
              "    robot.left_arm.delta_pose(0.1, 0.0, 0.0, 0.0, 0.0, 0.0)",
              f"    return skills.check_attached({obj}, robot.left_arm.gripper)"]
    elif name == "RECOVER":
        L += [f"    return skills.recover_grasp({obj})"]
    elif name == "VERIFY":
        L += [f"    {target} = skills.detect({_q(target)}, obs)",
              f"    return skills.verify_inside({obj}, {target})"]
    else:  # DONE / DETECT -> minimal valid chunk
        L += ["    return None"]
    return "\n".join(L) + "\n"
