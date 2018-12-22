from vpython import *
import h5py
import numpy as np

def numpy_to_vector(nparray):
    return vector(nparray[0], nparray[1], nparray[2])



with h5py.File('test.hdf5', 'r') as f:
    data = f['positions']
    spheres = []
    for i in range(data.shape[1]):
        spheres.append(simple_sphere(pos=vector(numpy_to_vector(data[0, i, :])), radius=0.05, make_trail=True, retain=100))
    for frame in data:
        rate(1000)
        for i, position in enumerate(frame):
            spheres[i].pos = numpy_to_vector(position)
