---
name: transport
description: RoboTwin transport — robot conventions, deriving reachable targets from perception, lift/approach heights, wrist-rotation reach recovery, and bimanual cross-body carries via handover.
---

# Transport — RoboTwin (includes bimanual coordination)

## Robot conventions every program follows

These are the control rules the library has accumulated; the API contract itself is in
`capsim/integrations/surfaces/` and the constitution.

- **Near-arm rule.** Pick the arm on the object's side of the table (`"left" if pos[0] < 0 else "right"`),
  from a perceived pose, never a hard-coded side. The exception is a destination only one arm can reach
  (see the reach-probe entry below).
- **Lift before horizontal travel.** Raise the held object by about 0.10 m or more before any horizontal
  move; a low carry drags it through neighbours.
- **`z_approach` on descents.** Use `z_approach=0.10` on grasps and placements so the arm arrives from
  above; ladder it down toward `0.0` only when the backed-off pose is out of reach (see [grasp.md](grasp.md)).
- **Home the idle arm before moving the other one across the table.** The two arms do not avoid each other;
  a parked arm over the table will be hit. The expert returns an arm home only when the other takes over.
- **Never open the gripper mid-air on failure.** Retry with a different orientation or target; if
  abandoning, lower to just above the table and release there. Dropping from height scatters the scene and
  usually loses the trial.
- **Free retries first.** A failed `goto_pose` plans without moving, so candidate ladders cost no frames;
  executed recovery motions cost frames and teach hesitation.

## Never invent table coordinates: derive targets from perceived poses, then clamp to the reach band

**Trigger** — the task says "centre of the table" / "next to X" and the program must pick absolute
coordinates. Symptom: objects placed off the table edge, or placements that never plan because the target
is at the far reach limit.

**Code**

```python
spawn_y = float(np.mean([info[n]["pos"][1] for n in NAMES]))   # perceived band
row_y = float(np.clip(spawn_y, REACH_Y_MIN, REACH_Y_MAX))        # both-arm reach band (robot property)
d01 = halves[0] + halves[1] + GAP                               # spacing from extents
d12 = halves[1] + halves[2] + GAP
x0 = -(d01 + d12) / 2.0                                         # row centred on x=0
```

**Why it works** — the objects spawn in the usable workspace, so their own poses define where "the centre"
is; the clamp keeps the target inside the band both arms can reach with a held object. Programs that invent
a row far toward the robot lose objects off the front edge; an unclamped mean can sit at the far-reach limit
where neither arm plans. Establish the band with a reach probe (below), not by guessing.

## "Move away / further outward" = increase |x| toward the object's own edge

**Trigger** — a task whose goal is to move an object "away from the table" or "further to the outward side".
The robot base sits at very negative y, so the outward sides are the ±x table edges; a toward-robot (−y)
move does not count as "away", and a radial-from-centre move is dominated by −y whenever |y| > |x| and
under-moves x.

**Code**

```python
pos, _ = get_object_pose("playing card")        # perceive the side, don't hard-code it
arm     = "left" if pos[0] < 0 else "right"      # near-arm rule
outward = -1.0   if pos[0] < 0 else 1.0          # push toward the object's OWN edge
# after grasp + lift (z += ~0.12):
for dist in [0.30, 0.26, 0.22, 0.20]:            # largest reachable step first, ladder down
    tgt = grasp_pos.copy()
    tgt[0] = grasp_pos[0] + outward * dist        # travel in x only; hold y at spawn
    tgt[2] = grasp_pos[2] + 0.02
    if goto_pose(arm, tgt, grasp_quat, z_approach=0.10):
        break
# if none plan, pull toward the robot for reach: dy in [-0.05, -0.10] × the dist ladder
```

**Why it works** — the object's own perceived x tells you which edge is "its" outward side; driving |x| up
in that direction (holding y) is what the task means.

## Reach recovery ladder: rotate the wrist ±90° before giving up on an arm

**Trigger** — `goto_pose` returns False for a placement that is geometrically inside the workspace.

**Code**

```python
attempts = [(quat, offset)]
for deg in (90.0, -90.0):                    # rotate about WORLD z …
    attempts.append((rotz(quat, deg), rotz_vec(offset, deg)))   # … and the offset too
for q_i, off_i in attempts:
    if goto_pose(hand, base + off_i, q_i, z_approach=Z_APPROACH):
        open_gripper(hand); break
```

**Why it works** — a different wrist yaw changes the IK branch and the finger footprint, recovering marginal
reach and clearance next to a neighbour; for near-cubic objects the object rotates with it harmlessly.
Rotate the in-hand offset by the same angle or the placement drifts.

## A fixed edge or floor container is reachable by ONE arm only — probe both, don't assume near-arm

**Trigger** — the destination is a container fixed at a far table or floor edge (it never varies across
seeds), and you must both acquire the source and deliver into this container. The near-arm rule picks the
arm by the source side, but the destination may be reachable by only one arm regardless of where the
source is.

**Code** — probe both arms at the destination over a small height ladder before committing, homing between
probes so a parked arm never blocks:

```python
DOWN = np.array([0.5, -0.5, 0.5, 0.5])          # mouth/tool pointing down
reach = {}
for h in ("left", "right"):
    reach[h] = any(
        goto_pose(h, np.array([dst[0], dst[1], zz]), DOWN) for zz in (0.95, 0.90, 0.85, 0.80)
    )
    goto_home_joint_position(h)
# pick the arm that reaches the destination; if only the far arm reaches the
# source, grasp with the near arm then hand over to the destination-capable arm.
```

**Why it works** — a planned-but-failed probe costs no frames, and it decides the arm before any object is
committed, so the program never strands an object in the wrong hand. A probe like this is also how to
establish the both-arm reach band used by the clamp entry above.

## Bimanual: pick with the near arm, home the idle arm, hand over through the front band

**Trigger** — a target sits on the far side for the arm holding the object. The two arms do not avoid
each other, so a parked arm over the table will be hit.

**Code**

```python
prefer = "left" if block_xy[0] < 0 else "right"     # near-arm rule
...
home(other(hand))                                    # before moving this arm
...
side = -1.0 if hand == "left" else 1.0               # handover: two spots,
for hx, hy in [(0.05 * side, row_y - 0.10),          #   giver-side first,
               (0.0, row_y - 0.16)]:                 #   deeper neutral fallback
    if place_held(name, hand, cmd_pos, quat, np.array([hx, hy]), z):
        g = grasp(name, other(hand))                 # receiving arm re-grasps
        ...
```

**Why it works** — the front band (more negative y, closer to the robot) is where both arms' workspaces
overlap, so an object parked there is re-graspable by either arm; the deeper second spot covers the case
where the receiver cannot reach the first. A handover costs a full gripper cycle, so prefer a layout where
both arms reach everything (a row in the shared band) and reserve the handover for tasks whose expert demo
also hands over. Mixing a row outside the shared band with no handover path is the one combination that
cannot work.

## Bimanual lift: raise a two-arm-held object by pure vertical translation, in small alternating steps

**Trigger** — both arms hold one rigid object and must raise it. Tempted to also translate it horizontally
(toward the robot, to buy an arm more reach). Don't: any horizontal motion under load rotates or slides a
marginal grip off its grasp point.

**Code**

```python
for i in range(STEPS):                            # e.g. 2 x 0.07 m rather than one 0.16 m lift
    lz[2] += dz; rz[2] += dz                     # BOTH arms, same vertical step
    rl = goto_pose("left",  lz, l_quat)
    rr = goto_pose("right", rz, r_quat)
    if not rl: lz[2] -= dz                        # a stalled arm just holds its height
    if not rr: rz[2] -= dz
```

**Why it works** — two arms holding one rigid object form a closed kinematic chain; a single large
one-arm lift breaks it and the object slips through the jaws even though both grips "held". Keep each step
well under the grip's compliance and alternate the arms rather than committing either one to a long
unilateral move. A reach ceiling is better left as an honest failure than papered over with a drift that
breaks the passing scenes.

## Height by "what is it standing on", not by the held object's geometry

**Trigger** — an object held in the gripper must end up resting on another object, and you cannot see the
held object's underside (so its height is unknowable).

**Code**

```python
# the tool's underside is on the table RIGHT NOW, so this rise puts it on the target face
rise = (target_top_z - observed_table_z) - PRESS
goto_pose(hand, grasp_pos + np.array([shift_xy[0], shift_xy[1], rise]), grasp_quat)
```

`observed_table_z` must be the depth-observed surface, never `centre - half_height` of a resting object:
RoboTwin seats objects a few millimetres sunk into the table, so the geometric bottom leaves exactly that
gap, enough to miss a contact condition outright.

**Why it works** — the difference of two observable heights cancels the unobservable one. The same
trick sets a held object down on a measured raised surface: after the lift, `offset = carried - held_pose`
has `offset[2]` equal to the object's grip height above its own base, so `surface_z + small clearance +
offset` (same grasp quaternion) lands the base on the surface whatever the object is.

## Fold the post-lift re-read into the NEXT target instead of correcting later

**Trigger** — any carry where the object may settle in the jaws. (Companion to grasp.md's "measure the
in-hand offset": this is what to do with the measurement.)

**Code**

```python
goto_pose(hand, grasp_pos + np.array([0, 0, LIFT]), grasp_quat)
lifted_centre, _ = get_object_pose(obj)
slip = np.asarray(lifted_centre) - (centre_before + np.array([0, 0, LIFT]))
feature_xy = feature_xy + slip[:2]        # zero frames, no corrective motion
```

**Why it works** — the correction is applied to a target you have not committed to yet, so it costs no
executed motion and cannot itself destabilise the grip.

## Extract a far-corner object by pulling toward centre before lifting high

**Trigger** — an object grasped at a far x/y corner cannot be lifted straight up (the high pose is out of
reach) even though the grasp itself planned. A staging pose nearer the workspace centre at moderate height
is reachable.

**Code**

```python
init_z = float(get_object_pose(name)[0][2])
for fx, dy, dz in [(1.0, -0.05, 0.09), (0.75, -0.10, 0.07), (0.60, -0.12, 0.06)]:
    lp = grasp_pos.copy()
    lp[0] = grasp_pos[0]*fx; lp[1] = grasp_pos[1]+dy; lp[2] = grasp_pos[2]+dz  # in toward centre+robot
    if goto_pose(hand, lp, grasp_quat):
        bp, _ = get_object_pose(name)
        if bp is not None and float(bp[2]) - init_z > 0.05:   # VERIFY it actually rose
            offset = lp - np.asarray(bp, float); break        # measured in-hand offset
```

**Why it works** — scaling x toward centre (`*fx`) plus a small toward-robot `dy` keeps the lifted pose
inside the reachable shell that a straight-up lift falls outside; the rise check also catches a
closed-but-empty gripper. Pairs with the `z_approach` ladder in [grasp.md](grasp.md).

## Recover an unreachable placement by walking INWARD ACROSS THE TARGET SURFACE, not by giving up

**Trigger** — the placement target is geometrically valid but `goto_pose` returns False for both arms at
every yaw and every `z_approach` (a reach map over the target finds nothing plannable), typically when the
object and target sit near the table centre.

**Code** — any point inside the *measured* surface footprint still lands the object on the surface. Order
candidates by distance from the centre, toward the robot first, and take the first that plans; every failed
candidate costs zero frames:

```python
def landing_points(centre_xy, half_extent, hand):
    span = np.clip(half_extent, 0.02, 0.06)              # the surface's own measured half-size
    side = -1.0 if hand == "left" else 1.0
    out = [np.asarray(centre_xy, float)]                 # true centre ALWAYS first
    for frac in (0.2, 0.35, 0.5, 0.65):                  # never past the footprint edge
        for step in ((0.0, -1.0), (side, 0.0), (-side, 0.0), (0.0, 1.0)):
            out.append(centre_xy + np.asarray(step) * frac * span)
    return out
```

**Why it works** — the workspace boundary runs almost along y, so a toward-robot step buys reach where a
sideways step does not; the walk is bounded by the perceived half-extent, never by a checker tolerance.
Provenance: place_object_scale (the centre-of-table scenes where the expert itself cannot plan).
