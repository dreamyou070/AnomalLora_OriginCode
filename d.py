import numpy as np

rand_np = np.random.rand(512, 512)
rand_np = np.expand_dims(rand_np, axis=2).repeat(3, axis=2)