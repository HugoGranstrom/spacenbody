import numpy as np
import h5py
import matplotlib.pyplot as plt

def kinetic_energy(data, masses):
    masses = masses.reshape(1, -1)
    v = np.linalg.norm(data[..., 3:6], axis=2)
    v_2 = v*v
    e_kin = 0.5*v_2*masses
    e_kin_total = np.sum(e_kin, axis=1)
    return e_kin_total

# only from the sun
def potential_energy(data, masses, GM):
    masses = masses.reshape(1, -1)
    # -GM*m/r, assume sun in barycenter
    r = np.linalg.norm(data[..., 1:, 0:3], axis=2)
    U = -GM[0]*masses[0, 1:]*r**-1
    U_total = np.sum(U, axis=1)
    return U_total

with h5py.File('cool_verlet.hdf5', 'r') as f:
    data = f["positions"]
    frame_1 = data[0]
    t = f["time"]
    masses = f["mass"][:]
    GM = f["GM"][:]
    n_bodies = frame_1.shape[0]
    r_v_frames = np.zeros((len(data), n_bodies, 2))
    """
    for i, frame in enumerate(data):
        for j in range(frame.shape[0]):
            r_v_frames[i, j, 0] = np.linalg.norm(frame[j, 0:3]) # radius
            r_v_frames[i, j, 1] = np.linalg.norm(frame[j, 3:6]) # velocity
    """
    r_v_frames[..., 0] = np.linalg.norm(data[..., 0:3], axis=2)
    r_v_frames[..., 1] = np.linalg.norm(data[..., 3:6], axis=2)
    
    e_kin = kinetic_energy(data, masses)
    e_pot = potential_energy(data, masses, GM)
    #plt.plot(t, potential_energy(data, masses, GM))
    #plt.plot(t, kinetic_energy(data, masses))
    #plt.plot(t, e_kin / e_pot) # should be 1/2
    print(np.mean(e_kin / e_pot))
    plt.show()


    
