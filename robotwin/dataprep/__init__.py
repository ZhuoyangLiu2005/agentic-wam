"""RoboTwin code-stream data: everything the trainer needs to consume datasets produced by codegen/sim.

The seam is one manifest per collected task (`dataset.yaml`, written by codegen/sim `scripts/make_dataset.py`):
a LeRobot v2.1 dataset plus the stage-slicing cache this package defines.

    slicing.py       stage-marker contract -> per-episode code slices, cycle cache, preamble (called by the collector
                     via $SLICER and by `prepare.py`; the deploy-time preamble for the hybrid eval comes from here too)
    text_cache.py    Cosmos-Reason1 prompt-embedding cache for every instruction a dataset carries
    data_config.py   one RobotVideoDataset block per manifest on top of data_base.yaml -> configs/robotwin/data/<name>.yaml
    smoke.py         instantiate a data config and fetch samples
    prepare.py       the three steps above in one command, plus a task config from task_template.yaml

Nothing here is imported by the trainer at run time; the trainer only reads the files these scripts emit.
Docs: docs/ROBOTWIN.md (data collection and training).
"""
