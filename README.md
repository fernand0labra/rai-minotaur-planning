# Minotaur Motion Planning - Ore Body Inspection


## The task, in short

Minotaur needs to autonomously explore an underground ore body: given a start
pose, an occupancy grid, a geological (ore prospectivity) map, and a goal, it
should generate waypoints roughly 50m ahead of the robot, inspect the ore
boundary as it goes, and eventually reach the goal. Four planners were asked
for: A*, Hybrid A*, RRT, RRT*, all driven by **one shared cost function**, so
comparisons between them are actually fair. The plan was always: get it working
in 2D first, extend to 3D, then visualize the result in Unity.

## Where we are

**2D: done, all four planners work.**
**3D: mostly done. A*, RRT, and RRT* work. Hybrid A* does not yet.**
**Unity: not started. This is the next big piece of work.**

Didnt have time to get the unity task to work due to accident

## Setup

```
# from the planning/ repo root
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`scipy` (used for RRT*'s KD-tree nearest-neighbor search) and `scikit-image`
(used for the 3D ore isosurface plot) should both be in `requirements.txt`.
Both planners still run without them, just slower / with a plainer plot (see
the notes near the bottom of this readme), so if either install gives you
trouble on a given machine it's not a blocker, just a quality tradeoff.

## The journey (why it's built this way)

We started purely in 2D: a flat ore field, a 2D occupancy grid, and the four
planners searching over `(x, y)`. That let us nail down the shared cost
function and the receding-horizon inspection logic without also fighting a 3D
search space.

Once 2D was solid, we merged the environment into one `Environment` class that
auto-detects 2D vs 3D from the `.npz` file (checks for a `field` key) and
generalized A*, RRT, and RRT* to work on either. Hybrid A* is the odd one out
because its state includes a heading angle `theta`, and "heading" in 3D isn't
a single angle, it needs a real orientation model (yaw + pitch, or a
quaternion), which is a genuine design decision, not a find-and-replace. That's
the next planner-side task, described more in the "what's left" section.

Unity visualization was always meant to be the last step, once the planners
were trustworthy in both dimensions. We're not there yet, right now the only
visual output is matplotlib (2D field + path plot, and a 3D isosurface plot).

## How the two repos fit together

This repo (`planning`) consumes maps produced by the **ore-body geological
digital twin** repo. That repo generates the ore field itself; this repo plans
paths through it. Rough flow:

```
(ore-body-digital-twin repo)
    python3 src/env2d.py                  -> a 2D .npz map (X, Y, Z)
    python3 src/env3d_smooth.py           -> a 3D .npz map (X, Y, Z, field)
    python3 src/env3d_voxel_occupancy.py  -> voxelized 3D occupancy variant
        |
        v
(this repo, planning/)
    Environment(npz_path) loads it, auto-detects 2D vs 3D
    python3 main.py map_path=<path to that .npz> planner=<astar|hybrid_astar|rrt|rrt_star>
```

If you're generating new maps, just point `map_path` at whatever `.npz` came
out of the digital-twin repo. The planning side doesn't care how the ore field
was generated, only that it has the right arrays (`x`, `y`, `X`, `Y`, `Z`, and
`field` for 3D).

### Generating the maps this repo actually uses

`config.py` defaults to `maps/ore_field_2d.npz`, and the 3D examples in this
readme use `maps/ore_field_3d.npz`. Neither of those files ships with this
repo, they come from the digital-twin repo and need to be generated and moved
over once, before you can run anything in 3D. From the digital-twin repo:

```
python3 src/env2d.py
python3 src/env3d_smooth.py
```

These will drop output `.npz` file(s) wherever that script writes them
(check the digital-twin repo's own readme/output path if you're not sure).
Then, from this repo's root, make sure `maps/` exists and copy them over with
the exact names this repo expects:

```
mkdir -p maps
cp /path/to/digital-twin-repo/<2d-output>.npz maps/ore_field_2d.npz
cp /path/to/digital-twin-repo/<3d-output>.npz maps/ore_field_3d.npz
```

After that, `map_path=maps/ore_field_2d.npz` and
`map_path=maps/ore_field_3d.npz` in the commands below will just work. If you
regenerate a map later (new seed, different bounds, etc), just re-run the
generator and overwrite the file in `maps/`, nothing else needs to change.

## Repo structure

```
planning/
  main.py                      entry point, builds env, runs inspection, plots
  config.py                    all tunables, CLI-overridable as key=value
  requirements.txt             python dependencies, see Setup above
  environment/
    environment.py             the shared world + cost map (2D and 3D)
  planners/
    astar.py                   A*        - works 2D + 3D
    rrt.py                     RRT       - works 2D + 3D
    rrt_star.py                RRT*      - works 2D + 3D
    hybrid_astar.py            Hybrid A* - works 2D only (see below)
    ore_waypoint.py            picks the next-best ore-boundary inspection point
  utils/
    priority_queue.py          tie-broken min-heap used by A* / Hybrid A*
    cost_utils.py               shared edge cost + collision check, used by every planner
  visualization/
    plotter.py                 2D plot: field + planned path
    plotter3d.py                3D plot: ore isosurface + planned path
```

## The shared cost function

This is the one thing every planner reads from, and it's the actual "brain" of
the inspection behavior. The planners themselves are just search strategies
on top of it:

```
cost(x, y[, z]) = w_obstacle    * (ore > obstacle_threshold ? 1 : 0)
                + w_prospectivity * ((ore - target_ore) / boundary_band)^2
                + w_uncertainty  * uncertainty(x, y[, z])
```

- The prospectivity term is a valley centered on `target_ore`, cheapest right
  on the ore boundary, expensive if you cut into the core or wander off into
  barren rock. That's what makes the robot hug the boundary instead of
  beelining for the goal.
- The uncertainty term starts at 1.0 everywhere and gets zeroed out in a
  radius around any waypoint the robot actually reaches (`env.reveal_around`).
  That's task item (b), "show it has more information of the surroundings"
  once it hits a goal location.
- The obstacle term is a hard wall above `obstacle_threshold`.

All four planners call the same `env.cost()` / `edge_cost()`, verified common
cost map, per spec.

## How the inspection route is generated

Early on we tried a fixed "orbit around the ore body" pattern (laps at
different tilt angles). That was scrapped, it assumes you know the ore body's
rough shape (a blob you can center and circle), which doesn't hold for
irregular or unknown geometries.

What's in `main.py` now instead: a **receding-horizon loop**. Each leg, it
asks the field itself (`ore_waypoint.py`) for the single most interesting
remaining point on the ore boundary, highest uncertainty, closest to
`target_ore`, plans to it, reveals uncertainty around it, and repeats. It
keeps going until either:

- the best remaining candidate isn't uncertain enough to bother with
  (`inspection_uncertainty_threshold`), or
- a safety cap on leg count is hit (`inspection_max_waypoints`, purely a
  backstop against a badly-tuned cost map looping forever),

and only then does it plan the final leg to the actual goal. This makes no
assumption about ore shape, it just chases whatever's still unexplored.

## Planner status in detail

| Planner      | 2D | 3D | Notes |
|--------------|----|----|-------|
| A*           | Yes | Yes | Grid search, dimension-agnostic from the start |
| RRT          | Yes | Yes | Sampling now uses numpy vectors, works for any dimension |
| RRT*         | Yes | Yes | Same, plus KD-tree (scipy) accelerated nearest-neighbor lookups |
| Hybrid A*    | Yes | No  | State is `(x, y, theta)`, 3D needs a real heading model |

**Why Hybrid A* is 2D-only right now:** its whole point is enforcing a turning
constraint (task item c) on top of the cost-driven search, and in 2D that's
just "yaw can change by at most `max_steer_deg` per step." In 3D, heading is
two angles (yaw + pitch), and the turning constraint becomes a cone rather
than an arc. It's not a huge lift, but it's a real change to the state
representation and the neighbor-generation logic, not a one-line fix. Worth
doing next if 3D Hybrid A* comparisons are needed for the final evaluation.

**Task item (c), turning constraints** (`theta in [-5, 5]` and
`theta in [-10, 10]` degrees): already supported, just a config value.

```
python3 main.py planner=hybrid_astar max_steer_deg=5
python3 main.py planner=hybrid_astar max_steer_deg=10
```

Both should be run and compared for the writeup, hasn't been done yet.

**Task item (d), robot length constraints:** not implemented anywhere yet.
Right now every planner treats the robot as a point mass. Adding a length
constraint would mean checking swept-volume collisions along a segment
(not just the endpoint), which touches `segment_collision_free` in
`utils/cost_utils.py` for every planner. Flagging this as fully open.

## Running it

Assuming `maps/ore_field_2d.npz` and `maps/ore_field_3d.npz` exist (see the
"Generating the maps" section above), here's every planner on every
environment it currently supports:

```
# ---- 2D (all four planners work) ----
python3 main.py map_path=maps/ore_field_2d.npz planner=astar
python3 main.py map_path=maps/ore_field_2d.npz planner=hybrid_astar
python3 main.py map_path=maps/ore_field_2d.npz planner=rrt
python3 main.py map_path=maps/ore_field_2d.npz planner=rrt_star

# ---- 3D (all except hybrid_astar, see "Planner status" above) ----
python3 main.py map_path=maps/ore_field_3d.npz planner=astar
python3 main.py map_path=maps/ore_field_3d.npz planner=rrt
python3 main.py map_path=maps/ore_field_3d.npz planner=rrt_star

# this one will currently crash, Hybrid A* isn't 3D-generalized yet:
# python3 main.py map_path=maps/ore_field_3d.npz planner=hybrid_astar
```

`planner=astar` is the default, so `map_path=...` alone runs A*.

Every field in `Config` (see `config.py`) is overridable as `key=value` on the
command line: cost weights, RRT/RRT* params, Hybrid A* steering, inspection
thresholds, all of it. Worth skimming `config.py` directly since it's the
single source of truth for every tunable. A couple of examples:

```
# different start/goal
python3 main.py map_path=maps/ore_field_2d.npz start_x=-60 start_y=-60 goal_x=60 goal_y=60

# tighter turning constraint on Hybrid A* (task item c)
python3 main.py map_path=maps/ore_field_2d.npz planner=hybrid_astar max_steer_deg=5
```

Output: `planned_path.npy` (for feeding into Unity later) plus a matplotlib
plot, field + planned path for 2D, ore isosurface + planned path for 3D
(needs `scikit-image` installed for the isosurface, falls back to a point
cloud if it's missing).

## What's left, roughly in priority order

1. **Unity visualization.** The big missing piece. `planned_path.npy` is
   already being saved specifically so it can be fed into Unity, that part
   should be straightforward, it's just never been wired up.
2. **Hybrid A* in 3D.** Needs the yaw+pitch heading extension described above.
3. **Robot length constraint (task item d).** Not started, currently
   everything is a point mass.
4. **Comparative evaluation.** Once Hybrid A* works in 3D, run all four
   planners on the same maps/start/goal and actually compare them (path cost,
   runtime, how thoroughly they inspect the boundary, behavior under both
   turning-constraint settings). This is explicitly asked for in the task
   definition and hasn't been done yet, everything so far has been about
   getting each planner to work correctly, not comparing them.
5. **Tuning pass on the inspection loop.** `inspection_uncertainty_threshold`
   and `inspection_reveal_radius_cells` in `config.py` control how many
   "laps"/legs the inspection does before heading to goal, worth revisiting
   per-map, since a good setting on one ore body/resolution won't necessarily
   be good on another.

## A couple of things worth knowing before you dig in

- `astar_heuristic_weight` / `hybrid_heuristic_weight` are "weighted A*" knobs,
  set above 1.0, they trade strict optimality for actually finishing in
  reasonable time. This is intentional: the spec's cost function creates cheap
  "moats" along the ore boundary that a plain (weight=1.0) heuristic can get
  stuck expanding forever. If a planner run isn't converging, this is usually
  the first knob to check.
- RRT*'s KD-tree acceleration needs `scipy` installed, without it, it falls
  back to brute-force nearest-neighbor search, which is dramatically slower
  and can look like it's hung when it's really just doing O(n^2) work. It'll
  print a warning on startup if scipy isn't found.
- All four planners return `(path, extra)` where `extra` holds
  `visited`/`tree` search data for debugging. `main.py` doesn't collect or
  plot this by default anymore (it got expensive across many inspection legs),
  if you need to see search internals again, that data is still there in
  each planner's return value, just not wired into the plotters currently.

That's the state of things. Good luck with it.
