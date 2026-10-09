"""Filled episode instructions reconciled with the program's arm: `{a}`-style arm placeholders record the EXPERT's
arm choice, so `reconcile` substitutes the program's first-grasp arm and regenerates the paraphrases (sim-free;
needs CAPX_ROBOTWIN_ROOT on sys.path)."""
import os
import sys
from pathlib import Path

ARMS = ("left", "right")


def program_hand(records):
    """Arm of the program's first grasp (close_gripper), else of its first motion; None if unknown."""
    for want in ("close_gripper", None):
        for r in records:
            if want and r.get("api") != want:
                continue
            h = r.get("hand") or (r.get("top_locals") or {}).get("hand")
            if h in ARMS:
                return h
    return None


def _robotwin_root():
    root = os.environ.get("CAPX_ROBOTWIN_ROOT") or str(Path(__file__).resolve().parents[1] / "third_party" / "RoboTwin")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def regenerate(task_name, placeholders, n=100):
    _robotwin_root()
    from description.utils.generate_episode_instructions import generate_episode_descriptions  # type: ignore
    out = generate_episode_descriptions(task_name, [placeholders], max_descriptions=n)
    d = out[0] if out else {}
    return {"seen": list(d.get("seen", [])), "unseen": list(d.get("unseen", []))}


def reconcile(task_name, doc, n=100):
    """-> (placeholders, instructions, changed: bool) for one alignment document."""
    ph = dict(doc.get("placeholders") or {})
    ins = doc.get("instructions") or {"seen": [], "unseen": []}
    hand = program_hand(doc.get("records") or [])
    arm_keys = [k for k, v in ph.items() if k.strip("{}").islower() and str(v) in ARMS]
    # only a single arm placeholder is decidable from the first grasp; handovers/bimanual keep the expert's
    if not hand or len(arm_keys) != 1 or str(ph[arm_keys[0]]) == hand:
        return ph, ins, False
    ph[arm_keys[0]] = hand                   # the program's arm is the truth for a single-arm task
    return ph, regenerate(task_name, ph, n), True
