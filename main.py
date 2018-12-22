import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import h5py
from tqdm import tqdm

import simulation

def run():
    config = simulation.ConfigLoader('config.json')
    positions, velocities, GM = config.load(('position','velocity','gm'))
    c = simulation.constants(1, 1, 1/(2-2**(1/3)), 0.1786178958448091, -0.2123418310626054, -0.6626458266981849E-01)
    s = simulation.Simulation(positions, velocities, GM, c)
    dt = -0.01
    total_time = -20000
    n_iter = int(total_time/dt)
    
    frames = np.zeros((n_iter, s.n, 3))
    start_time = time.time()
    for i in tqdm(range(n_iter)):
        frames[i, ...] = s.positions
        s.PEFRL_step(dt)
    print(f"Execution Time: {time.time() - start_time} seconds")
    s.save_positions_to_hdf5('test.hdf5', frames, 1)
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




