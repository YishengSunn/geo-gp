# Geometric Invariant Gaussian Process for 6D Trajectory Learning

This repository contains an interactive research codebase for **geometry-invariant one-shot Gaussian Process (GP) learning** of **6D (SE(3)) trajectories**.

The core idea is:

> Learn the **intrinsic geometric evolution** of a reference trajectory using geometry-invariant representations,  
> align a new probe trajectory to the reference via a similarity transform,  
> then **roll out predictions in the aligned reference frame** and map them back to the probe/world frame.

The system predicts **both position and orientation**, and the **probe trajectory orientation can differ from the reference orientation**.

## Features

- **One-shot GP learning** from a single reference trajectory
- **Geometry-invariant representation**
  - the **spherical coordinates of the reference trajectory** are used as the GP input
- **SE(3) trajectory prediction**
  - GP outputs **position and orientation deltas**
  - orientation is represented using **quaternions**
- **Autoregressive rollout prediction**
  - predicted deltas are applied step-by-step to generate the full trajectory
- **Similarity alignment** between probe and reference in 3D
- **Orientation-invariant prompting**
  - probe trajectory orientation can differ from the reference orientation
- **Skill library support**
  - automatically match the probe trajectory to the most similar reference trajectory
- **Interactive UI** for drawing trajectories, training models, and visualizing predictions

## Usage

First clone the repository and ensure that **uv** is installed.

```bash
git clone <repo-url>
cd <repo-directory>
```

Then run the application using:

```bash
uv run python3 -m main6d
```
