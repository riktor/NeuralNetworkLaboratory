import gnumpy as gpu
import numpy as np

n = 400

a = np.random.uniform(low=0., high=1., size=(n, n)).astype(np.float32)
b = np.random.uniform(low=0., high=1., size=(n, n)).astype(np.float32)

ga = gpu.garray(a)
gb = gpu.garray(b)

ga = ga.dot(gb)
a  = a.dot(b)

print ga.as_numpy_array(dtype=np.float32) - a
