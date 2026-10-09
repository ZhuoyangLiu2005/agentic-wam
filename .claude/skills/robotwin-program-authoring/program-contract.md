# Program contract for code-stream training (stage markers)

A frozen program is not only the collection oracle: its text is also what the policy's code expert is
trained to write, one **stage** at a time, and what the hybrid eval executes stage by stage. The stages are
declared inside the program with comment markers. `iterate/scripts/run_stage2.sh` checks them when it freezes the
program (a violation aborts the freeze); `pipeline/scripts/collect_aligned.sh` turns them into the training cache when
the collection ends. The engine is the trainer repo's slicer (`$SLICER`, see `env.sh`); its own documentation
is `$TRAINER_ROOT/docs/ROBOTWIN.md` §3.2.

## Markers (top-level statements of the main body)

    # [setup]                  perception / planning that runs once, before any motion (prelude context)
    # [stage k/N] <title>      the statements of stage k — one meaningful unit of the task, k = 1..N
    # [stage each] <title>     the next statement is a `for` over objects/hands; every executed iteration
                               becomes one stage; title may use the loop variables and UPPERCASE constants:
                               `# [stage each] pick and place {NAMES[idx]}`
    # [omit]                   audits / dead fallbacks that must not appear in the training text

Imports, UPPERCASE constants and `def`s are prelude material wherever they sit; sections may be reopened
(`# [setup]` again between two stages is fine). `print()` and `if ...: raise` guards are stripped
automatically, so keep them; they cost nothing. The one guard that is *not* stripped is one whose test
itself issues a motion (`if not reach(hand, p, q): raise`); see rule 11 for how to write it.

## Rules that shape how you write the main body

1. **Sequential phases.** Write the main body as a sequence of top-level phases in task order, each
   issuing at least one motion; mark each one. A stage boundary can only sit between top-level
   statements, never inside a loop body.
2. **Retry ladders stay inside a stage.** `for approach in (0.06, 0.1, 0.0): ... break` is part of the
   stage it serves, not a stage of its own.
3. **Object loops are `for` loops over a precomputed order.** If the task repeats one grasp-release cycle
   per object, compute the order in the setup (perception, conflict resolution, arm assignment) and
   iterate `for name in order:` / `for idx in PLACE_ORDER:` with plain names as loop targets, marked
   `# [stage each]`. Do not choose the next object inside a `while remaining:` loop: the iteration then
   has no loop variable, and the selection logic cannot be separated from the motion.
4. **Helper stubs need a return description.** Every helper the stages call gets a string return
   annotation `def press(hand, point) -> "(ok, lag): lag >= STALL_EPS means blocked":` (or a real type
   annotation, or a docstring whose first line says what comes back). The description documents the helper
   for readers and for the `--style desc` prompt; the default training prompt shows only the signature
   (`def press(hand, point): ...`) of the helpers the stages call, plus a `# Setup defines:` line naming the
   setup variables the stages read, so helper names must say what the phase does.
5. **No task knowledge outside the program.** Titles are the only free text; keep them short and
   task-semantic ("grasp the hammer by its shaft"). They become the stage headers the model generates.

## Slice shape (length and complexity)

A stage body is the text the code expert is trained to emit. It has to look like the code the model
already produces reliably: a short run of phase-level helper calls whose arguments are names, lookups and
UPPERCASE constants, plus a simple result check. The reference is the blocks_ranking_size stage body:

    name = NAMES[idx]
    h = move_block(name, slots[name], info[name]["pos"][2], row_y, last_hand)
    if h is None:
        failed.append(name)
    else:
        last_hand = h

A stage body describes *what the robot does to accomplish that step of the task*: which object, which arm,
which motion phase, in what order. The mechanics of finding a grasp, recovering from a failed plan,
assembling coordinates and carrying state from one phase to the next belong in helpers whose stubs the
policy sees. Two failure modes bracket the target:

- **Too long** — a raw retry ladder or a geometry computation left inline. A stage of several hundred
  tokens, dozens of lines and four or five indent levels is never emitted correctly from a masked canvas.
- **Plumbing** — multi-value unpacks, coordinate assembly and argument arithmetic at the call site. They
  carry no task semantics, the model has to memorise arities and argument orders, and they are where
  generation breaks.

Write each stage body to land between them:

6.  **40-128 tokens, 3-12 lines.** Sized against the eval decode canvas (the trainer's `infer_code_tokens`
    setting, 128 by default): a stage the canvas cannot hold is structurally unemittable no matter how good
    the model is. Check with the slicer and the trainer's tokenizer, not by eye.
7.  **Flat and sequential: one level of control flow.** Indent 0 is the norm. A stage body is a run of
    phase-level calls in task order, not a control-flow tree. One level is allowed where it carries the
    task's meaning: an `if`/`else` on a phase's result (`if h is None: ... else: ...`), or a `for` over a
    constant ladder or cycle count whose body is itself phase-level calls (`for quat in quat_ladder:`,
    `for i in range(SHAKE_CYCLES):`). A single `if` nested inside that block is tolerated; anything deeper,
    a `while`, a `try`, or a loop whose body searches rather than acts, belongs in a helper (see 9). Rule 2
    still holds: a ladder is never a stage of its own.
8.  **Put the intent in the stage title, not in body comments.** The slice text is produced by
    `ast.unparse`, and the Python AST does not retain comments, so every inline comment in a stage body is
    dropped. The only free text that reaches the model is the header line `# [stage k/N] <title>`, which
    the slicer prepends. Body comments are for the human reader; make the title carry the meaning
    ("grasp the hammer by its shaft", not "stage 1").
9.  **One helper per motion phase; the helper hides the search and the geometry.** `grasp_and_lift(...)`,
    `place_on_slot(...)`, `press(...)`: the call names the phase, the helper holds the candidate loop, the
    stand-off ladder, the coordinate assembly and the verification. Call arguments are plain names,
    dict/list lookups and UPPERCASE constants only: no inline numeric lists or tuples, no `np.array([...])`,
    no arithmetic or conditional expressions at the call site. Ladders and offsets live in UPPERCASE
    constants (hoisted into the preamble) or in helper defaults: `reach_grasp(hand, gp, gq, APPROACHES)`,
    not `reach_grasp(hand, gp, gq, [0.10, 0.05, 0.0])`.
10. **Records instead of wide returns.** A stage may unpack an API pair (`pos, quat = get_object_pose(name)`)
    or a two-value helper result; anything wider comes back as a dict record that the next phase takes
    whole (`grip = grasp_and_lift(...)`, then `place_on(grip, slots[name], rest_z)`). No chained assignments,
    comprehensions, lambdas or `x if c else y` in a stage body. Still banned as well: a bare magic number
    inside an expression, and any literal traceable to `check_success` (see the constitution).
11. **Guard on a bound result, never on the motion call itself.** The slicer drops an `if ...: raise`
    guard only when its test is free of motion and helper calls; `if not reach(hand, p, q): raise` is
    kept whole, `raise` included, because deleting it would delete the motion. Bind the motion first
    and guard the name on the next line:

        reached = reach(hand, grasp_pos, grasp_quat, Z_APPROACH)
        if not reached:
            raise RuntimeError("could not reach the shoe")

    The slice then keeps the motion and loses the guard, so the training text holds only the code that
    accomplishes the task; the guard still runs at Stage 2 and collection and aborts a failed episode
    before it can enter the dataset. The same applies to `if not goto_pose(...): raise`.

12. **`[stage each]` bodies read the same in every iteration.** The slicer substitutes the loop variables
    with that iteration's literals, so `if hand == "left":` becomes `if 'left' == 'left':`. Dispatch on the
    loop variable through data (`SIDE_X[hand]`, a precomputed order of tuples) or a helper, never with a
    comparison against the loop variable in the body.

Two consequences to be aware of before you refactor a passing program:

- **Moving a phase into a helper changes what eval tests.** The eval keeps full helper bodies and executes
  them regardless of what the model generated, so the ladder, the verification and the geometry inside a
  helper always run and stop being part of the code expert's score. That is the accepted trade-off of this
  style: the expert is scored on the phase sequence and its arguments, not on the mechanics; do not delete a
  ladder to "simplify".
- **Never park load-bearing work under `# [omit]`.** Omitted statements run during Stage 2 and collection but
  are excluded from both the preamble and the slices, so they never run at hybrid eval. The slicer's
  `omitted_motion` check only catches *direct* `goto_pose` / `close_gripper` / `open_gripper` /
  `goto_home_joint_position` calls; a motion reached through a helper called from an omitted statement
  passes the check and then silently never executes at eval.



## Tasks whose objects change per episode (program templates)

When the task draws its objects per episode, write each such object as the task's own instruction placeholder
inside a string literal — `locate("{A}")`, `grasp_and_lift(hand, "{B}", pos)` — using the keys of the task's
`info["info"]` rule (`envs/<task>.py`). At collection the key is replaced by that episode's noun, so the recorded
program and its slices name the object the instruction names. Rules: the keys appear only in stage bodies and
`[omit]` (never in `[setup]`, constants or helper defaults, which the model is given rather than writes); fixed
fixtures keep their literal noun; arm keys (`{a}`, `{b}`) are never used; comments must not contain a key.
Never refer to such an object by a generic name ("object", "target object"): the API does not ground those.

## Check before you hand the program over

    python3 $SLICER check <program.py>

It prints every main-body statement with its section, the stage list, and ERRORS for: missing or
non-contiguous stage numbers, a `[stage each]` not followed by exactly one `for` over plain names, a
slice or loop body that does not compile, an omitted statement that issues a motion.

Two engine rules worth knowing while you write: an UPPERCASE assignment is hoisted into the preamble's
constants block only when its value needs nothing from the main body (`Q = np.array(...)` yes,
`R0 = quat_matrix(mic_quat)` no); `if ...: raise` guards are stripped from the training text unless
their test itself calls a helper or a motion API, which is why rule 11 binds the motion to a name and
guards the name. A
`[stage each]` loop body may not contain `continue`/`return` (each iteration is emitted as straight-line code).

## Examples

Both are the main bodies of programs in `frozen/`, marked and sized as the rules above require.

### Sequential stages (a strike-with-tool task)

```python
# [stage 1/4] grasp the hammer by its shaft
# The arm on the block's side does the striking, so it gets first refusal; the
# far arm can only be a fallback because it cannot reach across the table.
hand = near_hand(block_centre)
grasp_pos, grasp_quat = reach_shaft_grasp(hand, 3, SLIDES, GRASP_STANDOFF)
if grasp_pos is None:
    hand = other_hand(hand)
    grasp_pos, grasp_quat = reach_shaft_grasp(hand, 3, SLIDES, GRASP_STANDOFF)
if grasp_pos is None:
    raise RuntimeError("could not reach a grasp on the hammer")
close_gripper(hand)

# [stage 2/4] lift it and measure how it settled in the jaws
# The hammer always slides a little in the jaws as it is picked up, and the
# strike aims with the HEAD, so the head's xy has to follow that slide.
lift = raised(grasp_pos, TRAVERSE_LIFT)
goto_pose(hand, lift, grasp_quat)
lifted_centre, _ = get_object_pose("hammer")
head_now_xy = head_after_slip(head_xy, hammer_centre, lifted_centre, TRAVERSE_LIFT)

# [stage 3/4] carry the head over the block
# The grip is not the head, so the grip moves by the head-to-block offset.
shift_xy = block_centre[:2] - head_now_xy
over = shifted(grasp_pos, shift_xy, TRAVERSE_LIFT)
if not goto_pose(hand, over, grasp_quat):
    carry_at_other_heights(hand, over, grasp_quat, CARRY_EXTRAS)

# [stage 4/4] lower the head onto the block and stop there
# The head has been resting ON THE TABLE, so lifting the whole tool by
# (block top - table surface) puts that same head on the block's top face.
flush_rise = block_top_z - table_z
press_rise = flush_rise - PRESS
land = shifted(grasp_pos, shift_xy, press_rise)
landed = goto_pose(hand, land, grasp_quat)
if not landed:
    landed = land_at_press_depths(hand, grasp_pos, grasp_quat, shift_xy, flush_rise, PRESS_DEPTHS)
```

### `[stage each]` over a precomputed order (an arrangement task)

```python
# [setup]
info = perceive(NAMES)
row_y, slots = row_slots(info, NAMES, JAW_CLEAR, REACH_BAND_Y)
rest_z = {n: info[n]["pos"][2] for n in NAMES}     # each block's resting height
order = tuple(NAMES[i] for i in PLACE_ORDER)
last_hand = None

# [stage each] move {name} onto its row slot
for name in order:
    g = grasp_near_arm(name, last_hand, GRASP_ROUNDS, Z_APPROACH, LIFT_LADDER)
    if g is not None:
        hand = finish_block(name, g, slots[name], rest_z[name], row_y)
        if hand is not None:
            last_hand = hand

# [omit]
for n in NAMES:
    p, q = get_object_pose(n)
    print("FINAL", n, None if p is None else np.round(p, 4).tolist())
```

Sections may be reopened: a `# [setup]` block between two stages, or a stage split around an `# [omit]`
audit, is fine as long as every statement of the stage issues its motions in task order.
