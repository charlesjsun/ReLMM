from softlearning.utils.serialization import (
    serialize_softlearning_object, deserialize_softlearning_object)

# from .random_network import RandomNetwork, rnd_predictor_and_target  # noqa: unused-import
# from .rnd_trainer import RNDTrainer  # noqa: unused-import

def serialize(value_function):
    return serialize_softlearning_object(value_function)


def deserialize(name, custom_objects=None):
    """Returns a random network function or class denoted by input string.

    Arguments:
        name : String

    Returns:
        Random network function function or class denoted by input string.

    For example:
    >>> softlearning.value_functions.get('rnd_predictor_and_target')
      <function rnd_predictor_and_target at 0x7f86e3691e60>
    >>> softlearning.value_functions.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown value function: abcd

    Args:
      name: The name of the random network function or class.

    Raises:
        ValueError: `Unknown value function` if the input string does not
        denote any defined random network.
    """
    return deserialize_softlearning_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='random network')


def get(identifier):
    """Returns a random network.

    Arguments:
        identifier: function, string, or dict.

    Returns:
        A random network function denoted by identifier.

    For example:

    >>> softlearning.value_functions.get('rnd_predictor_and_target')
      <function rnd_predictor_and_target at 0x7f86e3691e60>
    >>> softlearning.value_functions.get('abcd')
      Traceback (most recent call last):
      ...
      ValueError: Unknown value function: abcd

    Raises:
        ValueError: Input is an unknown function or string, i.e., the
        identifier does not denote any defined random network.
    """
    if identifier is None:
        return None
    if isinstance(identifier, str):
        return deserialize(identifier)
    elif isinstance(identifier, dict):
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError(
            f"Could not interpret random network function identifier:"
            " {repr(identifier)}.")
