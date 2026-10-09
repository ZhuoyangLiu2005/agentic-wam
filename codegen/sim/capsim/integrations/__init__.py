"""API registry. capsim registers the RoboTwin tiers only (see docs/robotwin-api-tiers.md)."""
from .base_api import list_apis, register_api  # noqa: F401

try:
    from .robotwin.base import RoboTwinBaseApi
    from .robotwin.base_privileged import (
        RoboTwinBasePrivilegedApi,
        RoboTwinGripperPrivilegedApi,
    )
    from .robotwin.gripper import RoboTwinGripperApi
except ImportError:
    print("RoboTwin not installed, skipping RoboTwin APIs")
else:
    register_api("RoboTwinBaseApi", RoboTwinBaseApi)
    # Perception API: Molmo/SAM3 grounding + GraspNet synthesis (honest stack).
    register_api("RoboTwinGripperApi", RoboTwinGripperApi)
    # Fully-privileged base surface: the unit-test oracle (boots no vision).
    register_api("RoboTwinBasePrivilegedApi", RoboTwinBasePrivilegedApi)
    # Full privilege (GT perception + GT reachability-validated grasp).
    register_api("RoboTwinGripperPrivilegedApi", RoboTwinGripperPrivilegedApi)
    # Tier 1: GT perception only — grasp synthesis stays real GraspNet.
    register_api(
        "RoboTwinGripperPrivilegedPerceptionApi",
        lambda env: RoboTwinGripperPrivilegedApi(
            env, privilege=frozenset({"perception"})
        ),
    )

    try:
        from .robotwin_trace_logger import (
            TracedRoboTwinGripperApi,
            TracedRoboTwinGripperPrivilegedApi,
        )
    except ImportError:
        pass
    else:
        register_api("RoboTwinGripperApiTraced", TracedRoboTwinGripperApi)
        register_api(
            "RoboTwinGripperPrivilegedApiTraced", TracedRoboTwinGripperPrivilegedApi
        )
        register_api(
            "RoboTwinGripperPrivilegedPerceptionApiTraced",
            lambda env: TracedRoboTwinGripperPrivilegedApi(
                env, privilege=frozenset({"perception"})
            ),
        )
