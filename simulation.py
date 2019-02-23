from collections import namedtuple
import numpy as np
import json
import h5py

constants = namedtuple("constants", "G c Theta Epsilon Lambda Chi")

class Simulation():
    # add __slots__
    def __init__(self, positions, velocities, GM, consts):
        # have separate positions arrays for bodies and comets. Maybe have two loops then. 
        self.positions = np.array(positions).reshape(-1, 3)
        self.velocities = np.array(velocities).reshape(-1, 3)
        
        self.data = np.concatenate((self.positions, self.velocities), axis=1)
        self.positions = self.data[..., 0:3]
        self.velocities = self.data[..., 3:6]

        self.accelerations = np.zeros(self.positions.shape)
        self.GM = np.array(GM).reshape(-1, 1) # make it a column vector
        assert self.positions.shape[0] == self.velocities.shape[0] == self.accelerations.shape[0] == self.GM.shape[0]
        self.n = len(self.GM)
        self.consts = consts
        self.index_except = np.array([[i for i in range(self.n) if i != a] for a in range(self.n)]) # index_except[i] gives all indices except i
        self.calculate_accelerations() # init accelerations for verlet
        #print(self.accelerations)

    def calculate_accelerations(self):
        self.accelerations *= 0 # reset accelerations
        for i in range(self.n):
            r_vectors = self.positions[self.index_except[i]] - self.positions[i]
            r_abs = np.sqrt((r_vectors*r_vectors).sum(axis=1)).reshape(-1, 1)
            accs = self.GM[self.index_except[i]] / r_abs**3 * r_vectors
            acc = accs.sum(axis=0) # add all the components
            self.accelerations[i] = acc

    def Euler_step(self, dt):
        self.calculate_accelerations()
        self.velocities += dt * self.accelerations
        self.positions += dt * self.velocities
    
    # not working proparly
    def Verlet_step(self, dt):
        self.positions += self.velocities * dt + self.accelerations * dt**2 / 2
        temp_acc = self.accelerations.copy()
        self.calculate_accelerations()
        self.velocities += dt/2 * (self.accelerations + temp_acc)
    
    def PEFRL_step(self, dt):
        self.positions += self.consts.Epsilon * dt * self.velocities
        self.calculate_accelerations()
        self.velocities += (1-2*self.consts.Lambda) * dt / 2 * self.accelerations
        self.positions += self.consts.Chi * dt * self.velocities
        self.calculate_accelerations()
        self.velocities += self.consts.Lambda * dt * self.accelerations
        self.positions += (1-2*(self.consts.Chi + self.consts.Epsilon)) * dt * self.velocities
        self.calculate_accelerations()
        self.velocities += self.consts.Lambda * dt * self.accelerations
        self.positions += self.consts.Chi * dt * self.velocities
        self.calculate_accelerations()
        self.velocities += (1-2*self.consts.Lambda) * dt / 2 * self.accelerations
        self.positions += self.consts.Epsilon * dt * self.velocities
    
    def save_positions_to_hdf5(self, filename, frames, t_frame, skip_length=1):
        # save positions matrices to hdf5 file
        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset("positions", data=frames[::skip_length, ...], compression="gzip")
            dset_t = f.create_dataset("time", data=t_frame[::skip_length], compression="gzip")
            dset_GM = f.create_dataset("GM", data=self.GM, compression="gzip")
            dset_mass = f.create_dataset("mass", data=self.GM/self.consts.G, compression="gzip")
        
    def calculate_error(self, end_positions, calc_positions):
        end_positions = np.array(end_positions).reshape(-1, 3)
        calc_positions = np.array(calc_positions).reshape(-1, 3)
        r_vectors = calc_positions - end_positions
        individual_error = np.sqrt((r_vectors**2).sum(axis=1))
        total_error = np.sum(individual_error)
        return total_error, individual_error

class ConfigLoader():
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'r') as f:
            config = json.load(f)
        self.simulation_params = config[1]
        self.planet_params = config[0]
        self.n_planets = len(self.planet_params)

    def load(self, *args):
        # check if multiple attributes are given
        if len(args) > 1:
            attributes = []
            for attr in args:
                # check if attr is string, else raise error
                if isinstance(attr, str):
                    attributes.append(self._load_attribute(attr))
                else:
                    raise ValueError(f"attr is of type {type(attr)}, needs to be a string")
            return tuple(attributes)
        elif len(args) == 1:
            return self._load_attribute(args[0])
        else:
            raise ValueError("No arguments where supplied, at least one is needed")

    def _load_attribute(self, attr):
        # one attribute
        if isinstance(attr, str):
            values = []
            for i, planet in enumerate(self.planet_params):
                if attr in planet:
                    values.append(planet[attr])
                else:
                    raise AttributeError(f"Planet {i} do not have an attribute '{attr}'")
            return values
        else:
            raise ValueError(f"Argument attr has type {type(attr)} but it has to be a string")
