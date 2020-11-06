import numpy as np

from collections import OrderedDict

from gym.spaces import Space

class DiscreteBox(Space):
    """ 
    A discrete action space, but each dicrete action is parameterized by a [a, b]^d vector,
    where d is a non-negative integer that can be different for each discrete action.

    Each discrete action can be named, or if not provided, default numbering will be used.

    Example: 

    >>> DiscreteBox(low=-1.0, high=1.0, dimensions=[2, 3, 0])
    DiscreteBox(OrderedDict([(0, 2), (1, 5), (2, 0)]))

    >>> DiscreteBox(low=0.0, high=1.0, 
                    dimensions={
                        "move": 2,
                        "grasp": 5, # 5 dof arm
                    })
    DiscreteBox(OrderedDict([('move', 2), ('grasp', 5)]))
    """
    def __init__(self, low=-1.0, high=1.0, dimensions=None, dtype=np.float32):
        assert dimensions != None
        assert isinstance(dimensions, (list, dict))
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = dtype
        
        if isinstance(dimensions, list):
            self.dimensions = OrderedDict(enumerate(dimensions))
        elif isinstance(dimensions, dict):
            self.dimensions = OrderedDict(dimensions)
        
        assert all(isinstance(d, int) and d >= 0 for d in self.dimensions.values()), 'dimensions must be non-negative integers'

        self.discrete_keys = list(self.dimensions.keys())

        self.low = self.dtype(low)
        self.high = self.dtype(high)
        
        self.num_discrete = len(self.dimensions)
        self.num_continuous = sum(self.dimensions.values())
        self.cumulative_dimension = np.cumsum(list(self.dimensions.values())).tolist()

    @property
    def shape(self):
        return self.dimensions

    def sample(self):
        """
        Chooses a random discrete action uniformly and then sample a random vector from the corresponding dimension.
        Returns a tuple of the action name and the vector.
        
        Example:
        
        >>> s = DiscreteBox(low=0.0, high=1.0, dimensions={"move": 2, "grasp": 1, "shutoff": 0})
        >>> s.sample()
        ("move", array([0.1132, 0.8323], dtype=float32))
        >>> s.sample()
        ("shutoff", None)
        """
        key = np.random.choice(self.discrete_keys)
        dim = self.dimensions[key]
        if dim == 0:
            return (key, None)
        else:
            sample = np.random.uniform(low=self.low, high=self.high, size=(dim,)).astype(self.dtype)
            return (key, sample)

    def contains(self, x):
        if not (isinstance(x, (list, tuple)) and len(x) == 2):
            return False
        key = x[0]
        sample = x[1]
        if key not in self.dimensions:
            return False
        dim = self.dimensions[key]
        if dim == 0:
            return sample is None
        sample = np.array(sample)
        if sample.shape != (dim,):
            return False
        return np.all(sample >= self.low) and np.all(sample <= self.high)

    def from_onehot(self, x):
        """ 
        Convert sample from a onehot encoding representation. 
        
        Example:

        >>> s = DiscreteBox(low=-1.0, high=1.0, dimensions=OrderedDict((("move", 2), ("grasp", 1), ("shutoff", 0))))
        >>> s.from_onehot(np.array([
            0.0, 1.0, 0.0,  # one-hot encoding for the 3 discrete actions ["move", "grasp", "shutoff"]
            -0.2, 1.0,      # "move" has 2 dims
            0.5,            # "grasp" has 1 dims
                            # "shutoff" has 0 dims
            ]))
        ("grasp", array([0.5], dtype=float32))
        """
        onehot = x[:self.num_discrete]
        i = np.argmax(onehot)
        key = self.discrete_keys[i]
        dim = self.dimensions[key]
        if dim == 0:
            return (key, None)
        sample = x[self.num_discrete + self.cumulative_dimension[i] - dim: self.num_discrete + self.cumulative_dimension[i]]
        return (key, np.array(sample, self.dtype))

    def to_jsonable(self, sample_n):
        return [[sample[0], np.array(sample[1]).tolist() if sample[1] is not None else None] for sample in sample_n]

    def from_jsonable(self, sample_n):
        return [(sample[0], np.asarray(sample[1]) if sample[1] is not None else None) for sample in sample_n]

    def __repr__(self):
        return f"DiscreteBox({self.dimensions})"

    def __eq__(self, other):
        return isinstance(other, DiscreteBox) and dict(self.dimensions) == dict(other.dimensions) \
            and np.isclose(self.low, other.low) and np.isclose(self.high, other.high)