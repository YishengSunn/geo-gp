"""Plot a synthetic chunk-wise rollout validation example."""

import numpy as np
import matplotlib.pyplot as plt

# Generate a simple circular reference trajectory.
t = np.linspace(0, 2 * np.pi, 200)
x_ref = np.cos(t)
y_ref = np.sin(t)

# Use the first part of the trajectory as the collected prompt.
prompt_end = 45

# Configure the predicted rollout chunks.
chunk_size = 30
num_chunks = 4

np.random.seed(1)
# Add light measurement noise to the collected prompt.
x_prompt = x_ref[:prompt_end] + 0.008 * np.random.randn(prompt_end)
y_prompt = y_ref[:prompt_end] + 0.008 * np.random.randn(prompt_end)

# Generate chunk predictions with continuous boundaries.
x_pred = x_ref.copy()
y_pred = y_ref.copy()
fail_start = prompt_end + chunk_size * 3
for i in range(num_chunks):
    start = prompt_end + i * chunk_size
    end = min(prompt_end + (i + 1) * chunk_size, len(t))
    if start >= len(t):
        break

    noise_scale = 0.006 if i < 3 else 0.01
    x_pred[start:end] = x_ref[start:end] + noise_scale * np.random.randn(end - start)
    y_pred[start:end] = y_ref[start:end] + noise_scale * np.random.randn(end - start)

    if i == 3:
        drift = np.linspace(0.0, 1.0, end - start)
        x_pred[start:end] += 0.46 * drift
        y_pred[start:end] -= 0.32 * drift

    prev_x = x_prompt[-1] if i == 0 else x_pred[start - 1]
    prev_y = y_prompt[-1] if i == 0 else y_pred[start - 1]
    x_pred[start:end] += prev_x - x_pred[start]
    y_pred[start:end] += prev_y - y_pred[start]

# Plot the reference, prompt, chunk rollout, and stop marker.
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(x_ref, y_ref, linestyle="--", linewidth=1.7, label="Reference")

ax.plot(
    x_prompt,
    y_prompt,
    linewidth=3.2,
    label="Collected Prompt"
)

for i in range(num_chunks):
    start = prompt_end + i * chunk_size
    end = min(prompt_end + (i + 1) * chunk_size, len(t))

    if start >= len(t):
        break

    passed = i < 3
    label = f"Chunk {i+1} {'✓' if passed else '✗'}"

    ax.plot(
        x_pred[start:end],
        y_pred[start:end],
        linewidth=2.4,
        label=label
    )

    mid = (start + end) // 2
    ax.text(
        x_pred[mid],
        y_pred[mid],
        f"C{i+1} {'✓' if passed else '✗'}",
        fontsize=12,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black")
    )

ax.scatter(
    x_pred[fail_start],
    y_pred[fail_start],
    s=120,
    marker="x",
    linewidths=3,
    label="Stop"
)

ax.set_title("Chunk-wise Rollout and Validation", fontsize=16)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.axis("equal")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

plt.tight_layout()
# plt.savefig("chunk_rollout_validation.png", dpi=300)
plt.show()
