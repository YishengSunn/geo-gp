import torch
import numpy as np

from config.runtime import K_HIST, METHOD_ID
from ui.app3d import App3D


def main():
    T = 100; t = np.linspace(0, 4*np.pi, T)
    radius = 1.0; speed = 0.1

    traj_train_np = np.stack([radius * np.cos(t), radius * np.sin(t), speed * t], axis=1)  # (T, 3)
    traj_train = torch.tensor(traj_train_np, dtype=torch.float32)

    traj_test_np = np.stack([speed * t, radius * np.cos(t), radius * np.sin(t)], axis=1)  # (T, 3)
    traj_test = torch.tensor(traj_test_np, dtype=torch.float32)
    
    INPUT_TYPE = 'spherical'  # 'pos', 'delta', 'pos+delta', 'spherical', 'spherical+delta', or 'dir'
    OUTPUT_TYPE = 'delta'     # 'delta' or 'absolute'

    start_t = K_HIST + 20                 # Start time for rollout
    h = traj_test.shape[0] - start_t - 1  # Rollout horizon
    n_align = 20                          # Number of points used for alignment

    app = App3D()
    app.set_reference(traj_train)
    app.set_probe(traj_test)
    app.train_reference_gp(k=K_HIST, input_type=INPUT_TYPE, output_type=OUTPUT_TYPE)
    app.estimate_alignment(n_align=n_align)
    app.rollout(start_t=start_t, horizon=h, input_type=INPUT_TYPE, output_type=OUTPUT_TYPE)
    app.plot(start_t=start_t)

if __name__ == "__main__":
    main()
