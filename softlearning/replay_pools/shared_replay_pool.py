import numpy as np
import tree

from .flexible_replay_pool import FlexibleReplayPool, Field, field_from_gym_space


class SharedReplayPool(FlexibleReplayPool):
    def __init__(self,
                 environment,
                 *args,
                 extra_fields=None,
                 **kwargs):
        extra_fields = extra_fields or {}
        observation_space = environment.observation_space
        action_space = environment.action_space

        self._environment = environment
        self._observation_space = observation_space
        self._action_space = action_space

        fields = {
            'observations': field_from_gym_space('observations', observation_space),
            'next_observations': field_from_gym_space('next_observations', observation_space),
            'actions': field_from_gym_space('actions', action_space),
            'rewards': Field(
                name='rewards',
                dtype='float32',
                shape=(1, )),
            # terminals[i] = a terminal was received at time i
            'terminals': Field(
                name='terminals',
                dtype='bool',
                shape=(1, )),
            'shared': Field(
                name='shared',
                dtype='bool',
                shape=(1, )),
            **extra_fields
        }

        super().__init__(
            *args, fields=fields, **kwargs)

        self.shared_indices = []

    def add_samples(self, samples):
        num_samples = tree.flatten(samples)[0].shape[0]

        assert (('episode_index_forwards' in samples.keys())
                is ('episode_index_backwards' in samples.keys()))
        if 'episode_index_forwards' not in samples.keys():
            samples['episode_index_forwards'] = np.full(
                (num_samples, *self.fields['episode_index_forwards'].shape),
                self.fields['episode_index_forwards'].default_value,
                dtype=self.fields['episode_index_forwards'].dtype)
            samples['episode_index_backwards'] = np.full(
                (num_samples, *self.fields['episode_index_backwards'].shape),
                self.fields['episode_index_backwards'].default_value,
                dtype=self.fields['episode_index_backwards'].dtype)

        index = np.arange(
            self._pointer, self._pointer + num_samples) % self._max_size

        def add_sample(path, data, new_values, field):
            assert new_values.shape[0] == num_samples, (
                new_values.shape, num_samples)
            data[index] = new_values
            if field.name == 'shared':
                for i in index:
                    if len(self.shared_indices) > 0 and i == self.shared_indices[0]:
                        del self.shared_indices[0]
                    if data[i][0]:
                        self.shared_indices.append(i)

        tree.map_structure_with_path(
            add_sample, self.data, samples, self.fields)

        self._advance(num_samples)
    
    @property
    def shared_size(self):
        return len(self.shared_indices)

    def random_batch_from_shared(self, batch_size):
        """ samples a random batch only from the shared data. """
        random_indices = np.random.choice(self.shared_indices, size=batch_size)
        return self.batch_by_indices(random_indices)

    def random_batch_from_both(self, batch_size, other_pool, process_other_batch=None):
        """ samples a random batch possibly including shared data from other_pool. """
        other_prob = other_pool.shared_size / (self.size + other_pool.shared_size)
        other_batch_size = np.random.binomial(batch_size, other_prob)
        if other_batch_size == 0:
            return self.random_batch(batch_size)
        other_batch = other_pool.random_batch_from_shared(other_batch_size)
        if process_other_batch:
            process_other_batch(other_batch)
        if other_batch_size == batch_size:
            return other_batch
        self_batch = self.random_batch(batch_size - other_batch_size)
        return tree.map_structure(lambda a, b: np.concatenate((a, b), axis=0), self_batch, other_batch)

    def random_batch_from_multiple(self, batch_size, other_pools, other_process_batches=None):
        """ samples a random batch possibly including shared data from other_pool. """
        total_size = np.sum([other_pool.shared_size for other_pool in other_pools]) + self.size
        sample_probs = [other_pool.shared_size / total_size for other_pool in other_pools] + [self.size / total_size]
        batch_sizes = np.random.multinomial(batch_size, sample_probs)
        
        batches = []

        if batch_sizes[-1] > 0:
            batches.append(self.random_batch(batch_sizes[-1]))

        for i in range(len(other_pools)):
            if batch_sizes[i] > 0:
                other_batch = other_pools[i].random_batch_from_shared(batch_sizes[i])
                if other_process_batches and other_process_batches[i]:
                    other_process_batches[i](other_batch)
                batches.append(other_batch)

        if len(batches) == 1:
            return batches[0]
        else:
            return tree.map_structure(lambda *b: np.concatenate(b, axis=0), *batches)


