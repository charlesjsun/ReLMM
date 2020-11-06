import numpy as np 
import tensorflow as tf
import tree

from scipy.special import expit

def create_grasping_env_discrete_sampler(
        env=None,
        discretizer=None,
        deterministic_model=None,
        min_samples_before_train=None,
        epsilon=None,
    ):
    total_dimensions = np.prod(discretizer.dimensions)

    def sample_random():
        obs = env.get_observation()
        action_discrete = np.random.randint(0, total_dimensions)
        action_undiscretized = discretizer.undiscretize(discretizer.unflatten(action_discrete))
        reward = env.do_grasp(action_undiscretized)
    
        return obs, action_discrete,reward, {'sample_random': 1, 'action_undiscretized': action_undiscretized}

    def sample_deterministic():
        obs = env.get_observation()   
        action_discrete = deterministic_model(np.array([obs])).numpy()
        action_undiscretized = discretizer.undiscretize(discretizer.unflatten(action_discrete))
        reward = env.do_grasp(action_undiscretized)

        return obs, action_discrete,  reward, {'sample_deterministic': 1, 'action_undiscretized': action_undiscretized}

    def sampler(num_samples, force_deterministic=False):
        if force_deterministic:
            return sample_deterministic()
        rand = np.random.uniform()
        if rand < epsilon or num_samples < min_samples_before_train: # epsilon greedy or initial samples
            #print("sampling random rand", rand, "epsilon", epsilon, "num_samples", num_samples, "minsamples", min_samples_before_train)
            return sample_random()
        else:
            #print("deterministic")
            return sample_deterministic()

    return sampler


def create_fc_grasping_env_discrete_sampler(
        env=None,
        discretizer=None,
        deterministic_model=None,
        min_samples_before_train=None,
        epsilon=None,
    ):
    total_dimensions = np.prod(discretizer.dimensions)

    def sample_random():
        obs = env.get_observation()
        action_discrete = np.random.randint(0, total_dimensions)
        #action_undiscretized = discretizer.undiscretize(discretizer.unflatten(action_discrete))
        reward = env.do_grasp(action_discrete)
    
        return obs, action_discrete, reward, {'sample_random': 1}

    def sample_deterministic():
        obs = env.get_observation()   
        action_discrete = deterministic_model(np.array([obs])).numpy()
        #action_undiscretized = discretizer.undiscretize(discretizer.unflatten(action_discrete))
        reward = env.do_grasp(action_discrete)

        return obs, action_discrete, reward, {'sample_deterministic': 1}

    def sampler(num_samples, force_deterministic=False):
        if force_deterministic:
            return sample_deterministic()
        rand = np.random.uniform()
        if rand < epsilon or num_samples < min_samples_before_train: # epsilon greedy or initial samples
            #print("sampling random rand", rand, "epsilon", epsilon, "num_samples", num_samples, "minsamples", min_samples_before_train)
            return sample_random()
        else:
            #print("deterministic")
            return sample_deterministic()

    return sampler


def create_fake_grasping_discrete_sampler(
        env=None,
        discrete_dimension=None,
        deterministic_model=None,
        min_samples_before_train=None,
        epsilon=None,
    ):
    def sample_random():
        obs = env.get_observation()
        action = np.random.randint(0, discrete_dimension)
        reward = env.do_grasp(action)
    
        return obs, action, reward, {'sample_random': 1}

    def sample_deterministic():
        obs = env.get_observation()   
        action = deterministic_model(np.array([obs])).numpy()
        reward = env.do_grasp(action)

        return obs, action, reward, {'sample_deterministic': 1}

    def sampler(num_samples):
        rand = np.random.uniform()
        if rand < epsilon or num_samples < min_samples_before_train: # epsilon greedy or initial samples
            return sample_random()
        else:
            return sample_deterministic()

    return sampler


def create_grasping_env_autoregressive_discrete_sampler(
        env=None,
        discretizer=None,
        deterministic_model=None,
        min_samples_before_train=None,
        epsilon=None,
    ):
    
    def sample_random():
        obs = env.get_observation()
        action_discrete = np.random.randint(0, discretizer.dimensions)
        action_undiscretized = discretizer.undiscretize(action_discrete)
        reward = env.do_grasp(action_undiscretized)

        return obs, action_discrete, reward, {'sample_random': 1}

    def sample_deterministic():
        obs = env.get_observation()
        action_onehot = deterministic_model(np.array([obs]))
        action_discrete = np.array([tf.argmax(a, axis=-1).numpy().squeeze() for a in action_onehot])
        action_undiscretized = discretizer.undiscretize(action_discrete)
        reward = env.do_grasp(action_undiscretized)

        return obs, action_discrete, reward, {'sample_deterministic': 1}

    def sampler(num_samples):
        rand = np.random.uniform()
        if rand < epsilon or num_samples < min_samples_before_train: # epsilon greedy or initial samples
            return sample_random()
        else:
            return sample_deterministic()

    return sampler


def create_grasping_env_ddpg_sampler(
        env=None,
        policy_model=None,
        unsquashed_model=None,
        action_dim=None,
        min_samples_before_train=1000,
        num_samples_at_end=50000,
        noise_std_start=0.5,
        noise_std_end=0.01,
    ):
    """ Linear annealing from start noise to end noise. """

    def sample_random():
        obs = env.get_observation()
        action = np.random.uniform(-1.0, 1.0, size=(action_dim,))
        reward = env.do_grasp(env.from_normalized_action(action))
    
        return obs, action, reward, {'action': action}

    def sample_with_noise(noise_std):
        obs = env.get_observation()

        noise = np.random.normal(size=(action_dim,)) * noise_std
        action_deterministic = policy_model(np.array([obs])).numpy()[0]
        action = np.clip(action_deterministic + noise, -1.0, 1.0)

        reward = env.do_grasp(env.from_normalized_action(action))
        
        infos = {
            'action': action, 
            'action_deterministic': action_deterministic, 
            'noise': noise, 
            'noise_std': noise_std,
        }

        if unsquashed_model:
            action_unsquashed = unsquashed_model(np.array([obs])).numpy()[0]
            infos['unsquashed_action'] = action_unsquashed 

        return obs, action, reward, infos

    def sampler(num_samples):
        if num_samples < min_samples_before_train: # epsilon greedy or initial samples
            return sample_random()
        
        noise_std = np.interp(num_samples, 
                              np.array([min_samples_before_train, num_samples_at_end]), 
                              np.array([noise_std_start, noise_std_end]))

        return sample_with_noise(noise_std)

    return sampler


def create_grasping_env_ddpg_epsilon_greedy_sampler(
        env=None,
        policy_model=None,
        unsquashed_model=None,
        Q_model=None,
        action_dim=None,
        min_samples_before_train=1000,
        epsilon=0.1,
    ):

    def sample_random():
        obs = env.get_observation()
        action = np.random.uniform(-1.0, 1.0, size=(action_dim,))
        reward = env.do_grasp(env.from_normalized_action(action))
    
        return obs, action, reward, {'action': action}

    def sample_deterministic():
        obs = env.get_observation()
        action = policy_model(np.array([obs])).numpy()[0]
        action = np.clip(action, -1.0, 1.0)
        reward = env.do_grasp(env.from_normalized_action(action))

        infos = {'action': action}

        if unsquashed_model:
            action_unsquashed = unsquashed_model(np.array([obs])).numpy()[0]
            infos['unsquashed_action'] = action_unsquashed
        if Q_model:
            logits = Q_model((np.array([obs]), np.array([action]))).numpy()[0]
            infos['Q_logits_for_action'] = logits

        return obs, action, reward, infos

    def sampler(num_samples):
        rand = np.random.uniform()
        if rand < epsilon or num_samples < min_samples_before_train: # epsilon greedy or initial samples
            return sample_random()
        else:
            return sample_deterministic()

    return sampler


def create_grasping_env_Q_greedy_sampler(
        env=None,
        Q_model=None,
        num_samples=256,
        num_samples_elites=16,
        num_samples_repeat=4,
        action_dim=None,
        min_samples_before_train=1000,
        epsilon=0.1,
    ):

    if isinstance(num_samples, int):
        num_samples = [num_samples] * num_samples_repeat
    elif isinstance(num_samples, (list, tuple)):
        if len(num_samples) == 2:
            num_samples = [num_samples[0]] + [num_samples[1]] * (num_samples_repeat - 1)
        elif len(num_samples) != num_samples_repeat:
            raise ValueError

    def sample_random():
        obs = env.get_observation()
        action = np.random.uniform(-1.0, 1.0, size=(action_dim,))
        reward = env.do_grasp(env.from_normalized_action(action))
    
        return obs, action, reward, {'action': action}

    def sample_deterministic():
        obs = env.get_observation()

        max_action_samples = None
        max_action_Q_values = None

        for i, n in enumerate(num_samples):
            if i == 0:
                action_samples = np.random.uniform(-1.0, 1.0, size=(n, action_dim))
            else:
                action_means = np.mean(max_action_samples, axis=0)
                action_stds = np.std(max_action_samples, axis=0)
                # print(action_means, action_stds)
                action_samples = np.random.normal(loc=action_means, scale=action_stds, size=(n, action_dim))
                action_samples = np.clip(action_samples, -1.0, 1.0)

            stacked_obs = np.array([obs] * n)
            Q_values = Q_model((stacked_obs, action_samples)).numpy()
            
            max_Q_inds = np.argpartition(Q_values.squeeze(), -num_samples_elites)[-num_samples_elites:]
            
            max_action_samples = action_samples[max_Q_inds, ...]
            max_action_Q_values = Q_values[max_Q_inds, ...]

        max_ind = np.argmax(max_action_Q_values)
        action = max_action_samples[max_ind, ...]
        Q_value = expit(max_action_Q_values[max_ind, ...][0])

        reward = env.do_grasp(env.from_normalized_action(action))

        infos = {
            'action': action,
            'max_Q_value': Q_value
        }
        # print(infos)

        return obs, action, reward, infos

    def sampler(num_samples):
        rand = np.random.uniform()
        if rand < epsilon or num_samples < min_samples_before_train: # epsilon greedy or initial samples
            return sample_random()
        else:
            return sample_deterministic()

    return sampler


def create_grasping_env_soft_q_sampler(
        env=None,
        discretizer=None,
        logits_models=None,
        min_samples_before_train=None,
        deterministic=False,
        alpha=10.0,
        beta=0.0,
        aggregate_func="min",
        uncertainty_func=None
    ):
    total_dimensions = np.prod(discretizer.dimensions)

    def sample_random():
        obs = env.get_observation()
        action_discrete = np.random.randint(0, total_dimensions)
        action_undiscretized = discretizer.undiscretize(discretizer.unflatten(action_discrete))
        reward = env.do_grasp(action_undiscretized)
    
        return obs, action_discrete, reward, {"sample_random": 1, "action": action_undiscretized}

    @tf.function(experimental_relax_shapes=True)
    def calc_probs(obs):
        obs = obs[tf.newaxis, ...]

        all_logits = [logits_model(obs) for logits_model in logits_models]

        all_Q_values = tf.nn.sigmoid(all_logits)

        min_Q_values = tf.reduce_min(all_Q_values, axis=0)
        mean_Q_values = tf.reduce_mean(all_Q_values, axis=0)
        max_Q_values = tf.reduce_max(all_Q_values, axis=0)

        if aggregate_func == "min":
            agg_Q_values = min_Q_values
        elif aggregate_func == "mean":
            agg_Q_values = mean_Q_values
        elif aggregate_func == "max":
            agg_Q_values = max_Q_values
        else:
            raise NotImplementedError()

        if uncertainty_func is None:
            uncertainty = tf.constant(0.0)
        elif uncertainty_func == "std":
            uncertainty = tf.math.reduce_std(all_Q_values, axis=0)
        elif uncertainty_func == "diff":
            uncertainty = max_Q_values - min_Q_values
        else:
            raise NotImplementedError()
        
        probs = tf.nn.softmax(alpha * agg_Q_values + beta * uncertainty, axis=-1)

        diagnostics = {
            "min_Q_values": tf.squeeze(min_Q_values),
            "mean_Q_values": tf.squeeze(mean_Q_values),
            "max_Q_values": tf.squeeze(max_Q_values),
        }

        return tf.squeeze(probs), tf.squeeze(agg_Q_values), diagnostics

    def sample_policy():
        obs = env.get_observation()

        probs, agg_Q_values, diagnostics = calc_probs(obs)
        probs = probs.numpy()
        agg_Q_values = agg_Q_values.numpy()
        diagnostics = tree.map_structure(lambda x: x.numpy(), diagnostics)
        
        if deterministic:
            action_discrete = np.argmax(agg_Q_values)
        else:
            action_discrete = np.random.choice(total_dimensions, p=probs)

        action_undiscretized = discretizer.undiscretize(discretizer.unflatten(action_discrete))
        reward = env.do_grasp(action_undiscretized)

        infos =  {
            "sample_policy": 1, 
            "action": action_undiscretized,
            "max_Q_value": diagnostics["max_Q_values"][action_discrete],
            "mean_Q_value":  diagnostics["mean_Q_values"][action_discrete],
            "min_Q_value": diagnostics["min_Q_values"][action_discrete],
        }

        return obs, action_discrete, reward, infos

    def sampler(num_samples, force_deterministic=False):
        # initial samples
        if num_samples < min_samples_before_train and not force_deterministic:
            return sample_random()
        else:
            return sample_policy()

    return sampler
