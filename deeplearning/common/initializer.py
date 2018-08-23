import numpy as np

def xavier(shape, prev_out_dim):
	return np.random.randn(*shape) / np.sqrt(prev_out_dim)

def he(shape, prev_out_dim):
	return np.random.randn(*shape) * np.sqrt(2/prev_out_dim)

def normal(shape, prev_out_dim=None):
	return np.random.randn(*shape)*0.1