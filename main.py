import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import h5py
import json
from tqdm import tqdm
import sys

import simulation

def run():
    config = simulation.ConfigLoader('config.json')
    positions, velocities, GM = config.load('position','velocity','gm')
    names = config.load('name')
    print(names)
    c = simulation.constants(4 * np.pi ** 2 / 365.2422 ** 2, 1, 1/(2-2**(1/3)), 0.1786178958448091, -0.2123418310626054, -0.6626458266981849E-01)
    s = simulation.Simulation(positions, velocities, GM, c)
    dt = float(sys.argv[1])
    total_time = float(sys.argv[2])
    simulation_name = sys.argv[3]
    save_every_n_iterations = 100
    n_iter = int(total_time/dt)
    effective_dt = total_time/n_iter
    n_frames = n_iter // save_every_n_iterations
    t_frame = np.zeros(n_frames)
    frames = np.zeros((n_frames, s.n, 6))
    start_time = time.time()
    for i in tqdm(range(n_iter)):
        if i % save_every_n_iterations == 0:
            j = int(i/save_every_n_iterations)
            frames[j, ...] = s.data
            t_frame[j] = i * effective_dt
        s.Verlet_step(dt)
    print(f"Execution Time: {time.time() - start_time} seconds")
    s.save_positions_to_hdf5(f'{simulation_name}.hdf5', frames, t_frame, 1)
    with open(f"{simulation_name}_names.json", 'w') as f:
        json.dump(names, f)
    """
    X = frames[::5, 0:2, 0]
    Y = frames[::5, 0:2, 1]
    Z = frames[::5, 0:2, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.autoscale(enable=False, axis="both")
    ax.set_xbound(-10, 10)
    ax.set_ybound(-10, 10)
    ax.set_zbound(-1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    sct, = ax.plot(X[0], Y[0], Z[0], 'o', markersize=2)
    def update_plot(i, x, y, z):
        sct.set_data(x[i], y[i])
        sct.set_3d_properties(z[i])
    ani = animation.FuncAnimation(fig, update_plot, frames=range(n_iter), fargs=(X, Y, Z), interval=1)
    plt.show()"""

if __name__ == "__main__":
    run()




