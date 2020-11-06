import numpy as np

class Discretizer:
    def __init__(self, sizes, mins, maxs):
        self._sizes = np.array(sizes)
        self._mins = np.array(mins) 
        self._maxs = np.array(maxs) 
        self._total_dimensions = np.prod(self._sizes)
        self._step_sizes = (self._maxs - self._mins) / self._sizes

    @property
    def dimensions(self):
        return self._sizes

    def discretize(self, action):
        centered = action - self._mins
        indices = np.floor_divide(centered, self._step_sizes)
        clipped = np.clip(indices, 0, self._sizes)
        return clipped

    def undiscretize(self, action):
        return action * self._step_sizes + self._mins + self._step_sizes * 0.5

    def flatten(self, action):
        return np.ravel_multi_index(action, self._sizes, order='C')

    def unflatten(self, index):
        return np.array(np.unravel_index(index, self._sizes, order='C')).squeeze()