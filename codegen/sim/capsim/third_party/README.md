# External dependencies

Follow the module's [setup guide](../../SETUP.md). Ordinary dependencies are listed in
`requirements.txt`, `requirements-data.txt` and `requirements-perception.txt` at the module root.
`scripts/setup_external.sh` handles the external source, CUDA builds, assets and local configuration.

| Checkout | Required for |
|---|---|
| `RoboTwin` | Simulation, replay and collection |
| `contact_graspnet_pytorch` | Perception-privileged and full-perception grasp sampling |
| `sam3` | Full-perception segmentation |

Checkouts, assets, weights and service logs are machine-local and git-ignored.
