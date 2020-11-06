import os
import random

import tensorflow as tf
import numpy as np


PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))


def set_seed(seed):
    seed %= 4294967294
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Using seed {seed}")


def get_host_name():
    try:
        import socket
        return socket.gethostname()
    except Exception as e:
        print("Failed to get host name!")
        return None


class RunningMeanVar:
    def __init__(self, eps=1e-6):
        self._mean = np.zeros((), np.float64)
        self._var = np.ones((), np.float64)
        self._count = eps
        self._eps = eps

    def update_batch(self, batch):
        """ https://github.com/openai/random-network-distillation/blob/f75c0f1efa473d5109d487062fd8ed49ddce6634/mpi_util.py#L200-L214 """
        batch_mean = np.mean(batch)
        batch_var = np.var(batch)
        batch_count = batch.shape[0]
        
        delta = batch_mean - self._mean
        total_count = self._count + batch_count

        new_mean = self._mean + delta * batch_count / total_count
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self._count * batch_count / total_count
        new_var = m_2 / total_count

        self._mean = new_mean
        self._var = new_var
        self._count = total_count

    @property
    def count(self):
        return self._count

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var + self._eps

    @property
    def std(self):
        return np.sqrt(self._var + self._eps)

class RunningVarFromZero:
    """ Calculates running variance, but assuming mean is always 0. """
    def __init__(self, eps=1e-12):
        self._var = np.ones((), np.float64)
        self._count = eps
        self._eps = eps

    def update_batch(self, batch):
        batch_var = np.sum(np.square(batch))
        batch_count = batch.shape[0]
        
        total_count = self._count + batch_count
        new_var = (self._var * self._count + batch_var) / total_count

        self._count = total_count
        self._var = new_var

    @property
    def count(self):
        return self._count

    @property
    def var(self):
        return self._var + self._eps

    @property
    def std(self):
        return np.sqrt(self._var + self._eps)
