# MINOTAUR Project - Planning Module

This repository includes the code for orchestrating the control, state and environment modules to navigate a simulated robotic platform between start and goal borehole positions.

## Long Horizon Planner System (T4.1)

The planning is defined in the _planner.py_ file through the class __Planner__. This class utilizes all simulation components to guide an agent through the environment between two positions, tracking the ore-body gradient, receiving sensor data, estimating drill core textures and interpolating surrounding voxel gradients for geomodeling purposes. The parameterization includes:

```
"ore_threshold": 0.8
"step_length": 0.075
"reveal_radius": 0.05
"goal_tolerance": 0.1
"max_steps": 100
```

```
src/planner.py          # High-Level Planning
src/plot.py             # Component Plotting
```

The _plot.py_ file generates a plot for the visualization of the start/goal boreholes, navigated path, drill core textures, ore-body gradient and steer/track directions.

```
data/planner.html       # Navigation
```

The script saves the simulation run data to _planner.npz_, necessary for the interactive visualization provided by the supervision module.

```
data/planner.npz        # Run Data
```