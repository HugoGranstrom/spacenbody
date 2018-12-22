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
    
    def Verlet_step(self, dt):
        self.positions += self.velocities * dt + self.accelerations * dt**2 / 2
        temp_acc = self.accelerations
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
    
    def save_positions_to_hdf5(self, filename, frames, skip_length=1):
        # save positions matrices to hdf5 file
        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset("positions", data=frames[::skip_length, ...])

class ConfigLoader():
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'r') as f:
            config = json.load(f)
        self.simulation_params = config[1]
        self.planet_params = config[0]
        self.n_planets = len(self.planet_params)

    def load(self, attribute_names):
        # check if multiple attributes are given
        if isinstance(attribute_names, tuple) or isinstance(attribute_names, list):
            attributes = []
            for attr in attribute_names:
                # check if attr is string, else raise error
                if isinstance(attr, str):
                    attributes.append(self._load_attribute(attr))
                else:
                    raise ValueError(f"attr is of type {type(attr)}, needs to be a string")
            return tuple(attributes)

        # one attribute
        elif isinstance(attribute_names, str):
            return self._load_attribute(attribute_names)
        # raise error if it is neither    
        else:
            raise ValueError(f"Argument is of type {type(attribute_names)}, needs to be a string or list/tuple of strings")

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
