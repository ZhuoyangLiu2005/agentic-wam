# Collection checklist for agents

Use this runbook to turn a validated frozen program into a reviewed dataset. It defines execution
order and completion checks; commands and CLI behavior live in the
[user guide](../../../docs/ROBOTWIN.md).
Read [project rules](../../../codegen/sim/CLAUDE.md) first. Artifact schemas belong to the
[handoff contract](../../../docs/ROBOTWIN.md).

## 1. Establish the inputs

- Read the current `frozen/<task>/task.yaml`, original validation report and validated program/hash.
  Confirm the completed validation meets the project's gate; successful promotion is not approval.
- Check the [program contract](../robotwin-program-authoring/program-contract.md). Any changed program, including an override,
  requires validation review before collection.
- Complete [environment setup](../../../docs/ROBOTWIN.md).
  Confirm assigned GPUs, available disk space and downstream export/slicing dependencies.

## 2. Execute collection

Follow the [Stage B procedure](../../../docs/ROBOTWIN.md):
plan attempted seeds, choose the protocol and output directory, configure instruction recording,
and launch collection. Assign GPUs explicitly; the planner's host column does not allocate them.
Use one host per collection directory and separate outputs for protocols and program versions.

For interrupted work, inspect existing artifacts before using skip options. Preserve logs and the
program identity needed to establish which execution produced each dataset.

## 3. Review before handoff

- Check `dataset.yaml` for the actual kept count and usable slice paths.
- Inspect raw manifest success flags, requested/scene seeds and episode identifiers.
- Inspect successful alignments for matching embedded code, records, instruction metadata and errors.
- Check relevant `top_line` / `top_locals` for loop stages and nonempty stage-frame slices.
- Inspect exported samples for expected instructions, episode counts and colors.

Investigate low success rates, placeholder errors and empty stages before declaring completion.
Expert instruction mode is not a universal repair for placeholder failures.

**Done when:** the requested kept-episode target is met, required exports and slices are usable,
and the review above passes. Report output paths, actual counts and any unresolved limitations.
For downstream preparation and task import, follow the
[user guide](../../../docs/ROBOTWIN.md). The human reviews and pushes.
