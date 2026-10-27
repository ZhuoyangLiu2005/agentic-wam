"""hybrid_eval — step-level hybrid (generated code x WAM) evaluation on RoboTwin.

config.py   site / evaluation-config / task-asset loading (configs/robotwin/)
assemble.py header parse, repair, whitelist, prelude (shared by the arbiter and offline tools)
arbiter.py  HybridArbiter, hooked into robotwin/deploy/deploy_policy.py under AGENTICWAM_HYBRID=1
vision.py   start / status / stop of the perception servers
launch.py   one eval arm (hybrid or pure WAM), writes run_meta.json beside the results
rollout_viewer.py  build | serve the HTML viewer of those results (simulation runs only)
(docs: docs/ROBOTWIN.md)
"""
