"""Coding-agent prompt for policy-code generation — dual-arm aware.

This is the prompt-engineering fix for the dual-arm gap surfaced by the capability
report: RoboTwin instructions frequently name an arm ("with the left arm"), ask for
both ("use dual arms"), or involve two objects/handover. The earlier single-arm
prompt+API made those uncompilable. Here we (1) expose a dual-arm API
(robot.left_arm / robot.right_arm, matching the dual-arm `MockRobot`), and (2) give
the model explicit ARM-SELECTION rules so it picks the arm(s) from the instruction.

Shared by the code-expert report and the Claude-generated GT (both tools since removed) so
they used one canonical contract.
"""
from __future__ import annotations

DUAL_ARM_SPEC = """You write short-horizon robot policy code (code-as-policy) for a DUAL-ARM tabletop robot.
Output ONLY a Python function `policy_chunk(obs, state, robot, skills)` — no prose, no markdown fences.

The robot has TWO arms. Each arm is independent:
  robot.left_arm   robot.right_arm
  arm.gripper                                  # arm.gripper.closed
  arm.move_to(pose)                            # pose.pregrasp, pose.grasp
  arm.open_gripper(); arm.close_gripper()
  arm.delta_pose(dx,dy,dz,droll,dpitch,dyaw); arm.execute_trajectory(traj)

ARM SELECTION — choose the arm(s) from the instruction:
  - "left arm" / "with the left"            -> use robot.left_arm
  - "right arm" / "with the right"          -> use robot.right_arm
  - "dual arms" / "both arms" / "two <X>" / "handover" / one object per side
                                            -> use BOTH arms (assign one object/target per arm,
                                               or hand the object from one arm to the other)
  - unspecified                             -> default to robot.right_arm

Skills (arm-agnostic):
  skills.detect(name, obs) -> obj ; skills.track(name, obs) -> obj
  skills.sample_grasp(obj) -> pose ; skills.sample_place_pose(obj, target) -> pose
  skills.place_inside(obj, container) -> pose
  skills.check_attached(obj, arm.gripper) -> bool ; skills.verify_inside(obj, container) -> bool
  skills.recover_grasp(obj) ; skills.visual_diff(before, after)

Use explicit control flow: check preconditions, act, verify, and recover. Call only the APIs above."""

FEWSHOT = """EXAMPLE 1 — single arm, explicit side
Instruction: Use the left arm to grab the green bottle and lift it.
def policy_chunk(obs, state, robot, skills):
    arm = robot.left_arm
    bottle = skills.detect("green bottle", obs)
    if not skills.check_attached(bottle, arm.gripper):
        g = skills.sample_grasp(bottle)
        arm.move_to(g.pregrasp); arm.move_to(g.grasp); arm.close_gripper()
        if not skills.check_attached(bottle, arm.gripper):
            return skills.recover_grasp(bottle)
    arm.delta_pose(0.0, 0.0, 0.15, 0.0, 0.0, 0.0)
    return skills.check_attached(bottle, arm.gripper)

EXAMPLE 2 — dual arms, one object per arm
Instruction: Use dual arms to place the hamburger and the can onto the plate.
def policy_chunk(obs, state, robot, skills):
    plate = skills.detect("plate", obs)
    jobs = [(robot.left_arm, skills.detect("hamburger", obs)),
            (robot.right_arm, skills.detect("can", obs))]
    for arm, obj in jobs:
        if not skills.check_attached(obj, arm.gripper):
            g = skills.sample_grasp(obj)
            arm.move_to(g.pregrasp); arm.move_to(g.grasp); arm.close_gripper()
            if not skills.check_attached(obj, arm.gripper):
                return skills.recover_grasp(obj)
        arm.move_to(skills.sample_place_pose(obj, plate)); arm.open_gripper()
    return skills.verify_inside(jobs[0][1], plate) and skills.verify_inside(jobs[1][1], plate)"""


def build_codegen_prompt(instruction: str) -> str:
    """Full single-shot prompt: spec + few-shot + the target instruction."""
    return f"{DUAL_ARM_SPEC}\n\n{FEWSHOT}\n\nInstruction: {instruction}\nWrite policy_chunk now."
