from matplotlib.animation import PillowWriter
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
num_cells = 50
num_lnp = 200
box_size = 10
uptake_radius = 0.5
uptake_prob = 0.05
steps = 200

# Initialize cell positions
cell_positions = np.random.rand(num_cells, 2) * box_size

# Initialize LNP positions and uptake status
lnp_positions = np.random.rand(num_lnp, 2) * box_size
lnp_uptaken = np.zeros(num_lnp, dtype=bool)

# Random walk parameters for LNPs
lnp_step_size = 0.2

def update_lnp_positions():
    # Random walk for LNPs that are not uptaken
    mask = ~lnp_uptaken
    if not np.any(mask):
        return
    directions = np.random.randn(num_lnp, 2)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    lnp_positions[mask] += directions[mask] * lnp_step_size
    # Keep within box
    lnp_positions[:] = np.clip(lnp_positions, 0, box_size)
    lnp_positions[:] = np.clip(lnp_positions, 0, box_size)

def check_uptake():
    for i, lnp_pos in enumerate(lnp_positions):
        if lnp_uptaken[i]:
            continue
        # Check distance to all cells
        dists = np.linalg.norm(cell_positions - lnp_pos, axis=1)
        if np.any(dists < uptake_radius):
            if np.random.rand() < uptake_prob:
                lnp_uptaken[i] = True

# Animation setup
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
cells_plot, = ax.plot([], [], 'go', markersize=10, label='Cells')
lnp_plot, = ax.plot([], [], 'bo', markersize=4, label='LNPs')
uptaken_plot, = ax.plot([], [], 'ro', markersize=4, label='Uptaken LNPs')
ax.legend(loc='upper right')

def init():
    cells_plot.set_data(cell_positions[:,0], cell_positions[:,1])
    lnp_plot.set_data([], [])
    uptaken_plot.set_data([], [])
    return cells_plot, lnp_plot, uptaken_plot

def animate(frame):
    update_lnp_positions()
    check_uptake()
    lnp_plot.set_data(lnp_positions[~lnp_uptaken,0], lnp_positions[~lnp_uptaken,1])
    uptaken_plot.set_data(lnp_positions[lnp_uptaken,0], lnp_positions[lnp_uptaken,1])
    return cells_plot, lnp_plot, uptaken_plot

ani = animation.FuncAnimation(fig, animate, frames=steps, init_func=init,
                              interval=50, blit=True, repeat=False)
ani.save('lnp_uptake_simulation.gif', writer=PillowWriter(fps=30), dpi=100)
plt.title("Animated Simulation of LNP Uptake")
plt.show()
