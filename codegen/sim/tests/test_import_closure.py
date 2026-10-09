"""capsim must import on its own: no `aspire`, no LIBERO / robosuite / franka / r1pro module may end up loaded.
Run with the RoboTwin python: source env.sh && $ROBOTWIN_PY tests/test_import_closure.py"""
import importlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
MODULES = ["capsim", "capsim.envs.launch", "capsim.envs.runner", "capsim.envs.trial", "capsim.envs.simulators.robotwin",
           "capsim.envs.tasks.robotwin.robotwin_env", "capsim.integrations", "capsim.integrations.robotwin_trace_logger",
           "capsim.integrations.robotwin.code_alignment", "capsim.utils.launch_utils", "capsim.data.hdf5", "capsim.profiles"]
for m in MODULES:
    importlib.import_module(m)
foreign = sorted(m for m in sys.modules if m.split(".")[0] == "aspire" or m.startswith(("libero", "capsim.envs.simulators.robosuite", "capsim.integrations.franka", "capsim.integrations.r1pro")))
assert not foreign, f"foreign modules loaded: {foreign}"
from capsim.integrations import list_apis
apis = list_apis(); assert "RoboTwinGripperPrivilegedApiTraced" in apis and all(a.startswith("RoboTwin") for a in apis), apis
print(f"ok: {len(MODULES)} modules, {len(apis)} RoboTwin APIs, no foreign module")
