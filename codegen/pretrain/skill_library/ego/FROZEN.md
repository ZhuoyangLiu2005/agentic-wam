# ego skill library — frozen

This library holds **27 callable skills** composed over the ego embodiment's
declaration-only API (`EgoHandApi`), written at SUB-TASK altitude: one skill = one
annotated sub-task slice, with its hand (contact) contract derived by expanding the
body. The library is FROZEN: editing it changes the prompt version and
invalidates the skeleton store, so re-run the skeleton `generate` stage before
`assemble`.

Skills:

- align-and-seat-braced
- brace-and-pick
- brace-and-take-tool
- close-closure
- flatten-by-pulling-taut
- flip-over-between-hands
- fold-over-and-crease
- hang-item-on-support
- insert-into-receptacle
- move-item-two-handed
- open-closure
- pick-and-bring
- place-at-target
- pour-into-target
- press-onto-target
- pull-open-and-hold
- pull-part-free
- rotate-grasped-object
- scoop-and-carry
- scrub-inside-vessel
- stir-contents
- thread-through-and-pull-out
- tighten-with-ratchet-regrasps
- turn-over-object
- unscrew-cap
- wipe-surface-back-and-forth
- wrap-around-target
