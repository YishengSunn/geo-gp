# Geometric Invariant Gaussian Process for 3D / 6D Trajectory Learning

This repository contains an interactive research codebase for **geometry-invariant one-shot Gaussian Process (GP) learning** of **3D and 6D (SE(3)) trajectories**.

The core idea is:

> Learn the **intrinsic geometric evolution** of a reference trajectory using geometry-invariant representations,  
> align a new probe trajectory to the reference via a similarity transform,  
> then **roll out predictions in the aligned reference frame** and map them back to the probe/world frame.

## Features

- **One-shot GP learning** from a single reference trajectory
- **Geometry-invariant representations**
  - planar angle representations relative to a start tangent
  - 3D trajectory inputs (e.g., spherical features)
  - 6D pose handling (position + orientation in SO(3))
- **Similarity alignment** between probe and reference in 3D
- **Rollout in the reference frame**, then projection back to probe/world
- **6D pipeline support** (orientation visualization + SO(3) log/exp utilities)
- **Interactive UI** to draw, train, align, predict, and visualize results
