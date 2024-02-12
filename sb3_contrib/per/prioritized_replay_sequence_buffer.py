# This will subclass the Prioritised Experience Replay
# and incorporates some of the functionality with HER
# to record episodic information within the buffer
# This currently only does non-prioritised samples

import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces

from sb3_contrib.per.prioritized_replay_buffer import PrioritizedReplayBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

class PrioritizedReplaySequenceBuffer(PrioritizedReplayBuffer):
    """
    This provides a Prioritized Replay Buffer that returns episodic sequences of transitions.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param alpha: How much prioritization is used (0 - no prioritization, 1 - full prioritization)
    :param beta: To what degree to use importance weights (0 - no corrections, 1 - full correction)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        optimize_memory_usage: bool = False,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, alpha, beta, optimize_memory_usage)

        # Episodic params from HER
        self.ep_start = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.ep_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self._current_ep_start = np.zeros(self.n_envs, dtype=np.int64)

        # LSTM state information
        # a 2-cell tuple of tensors
        self.lstm_states = (
                    np.zeros((self.buffer_size, self.n_envs), dtype=np.float32),
                    np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        )

    # from PER
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        lstm_states = None # for the moment whilst we replay whole episodes. othwerwise Tuple[np.ndarray]
    ) -> None:
        """
        Add a new transition to the buffer.

        :param obs: Starting observation of the transition to be stored.
        :param next_obs: Destination observation of the transition to be stored.
        :param action: Action performed in the transition to be stored.
        :param reward: Reward received in the transition to be stored.
        :param done: Whether the episode was finished after the transition to be stored.
        :param infos: Eventual information given by the environment.
        """
        # store transition index with maximum priority in sum tree
        # self.count should be self.pos??
        self.tree.add(self.max_priority, self.count)

        # from HER
        # When the buffer is full, we rewrite on old episodes. When we start to
        # rewrite on an old episodes, we want the whole old episode to be deleted
        # (and not only the transition on which we rewrite). To do this, we set
        # the length of the old episode to 0, so it can't be sampled anymore.
        for env_idx in range(self.n_envs):
            episode_start = self.ep_start[self.pos, env_idx]
            episode_length = self.ep_length[self.pos, env_idx]
            if episode_length > 0:
                episode_end = episode_start + episode_length
                episode_indices = np.arange(self.pos, episode_end) % self.buffer_size
                self.ep_length[episode_indices, env_idx] = 0

        ## NOTE: We will also want to update the priorities of these deleted transitions
        ## ie, update the deleted tree indices with the max_priority

        # Update episode start
        self.ep_start[self.pos] = self._current_ep_start.copy()

        # Store the transition
        # the parent class is non-RNN aware.  At the moment it doesn't need to be RNN aware
        super().add(obs, next_obs, action, reward, done, infos)
        #_add(obs, next_obs, action, reward, done, infos, lstm_states)

        # When episode ends, compute and store the episode length
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                self._compute_episode_length(env_idx)

    # the following is an overridden version of the inherited add() method
    # frome the ReplayBuffer parent
    def _add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        lstm_states = None  # for the moment.  Tuple[np.ndarray]
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        # At the moment, we don't store the lstm_states in the replay buffer
        # as we replay the entire episode
        # if we use randomised replay, we may wish to store this
        # altho the original paper zerod this state (test for performance loss in storing/retrieving this state)
        #self.lstm_states[self.pos] = (np.array(lstm_states[0]).copy(),np.array(lstm_states[1]).copy())

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _compute_episode_length(self, env_idx: int) -> None:
        """
        Compute and store the episode length for environment with index env_idx

        :param env_idx: index of the environment for which the episode length should be computed
        """
        episode_start = self._current_ep_start[env_idx]
        episode_end = self.pos
        if episode_end < episode_start:
            # Occurs when the buffer becomes full, the storage resumes at the
            # beginning of the buffer. This can happen in the middle of an episode.
            episode_end += self.buffer_size
        episode_indices = np.arange(episode_start, episode_end) % self.buffer_size
        self.ep_length[episode_indices, env_idx] = episode_end - episode_start
        # Update the current episode start
        self._current_ep_start[env_idx] = self.pos

    # adapted from HER
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample. Not used. We only return 1 episode.
        :param env: Associated VecEnv to normalize the observations/rewards when sampling
        :return: Samples
        """
        # When the buffer is full, we rewrite on old episodes. We don't want to
        # sample incomplete episode transitions, so we have to eliminate some indexes.
        is_valid = self.ep_length > 0
        if not np.any(is_valid):
            raise RuntimeError(
                "Unable to sample before the end of the first episode. We recommend choosing a value "
                "for learning_starts that is greater than the maximum number of timesteps in the environment."
            )
        # Get the indices of valid transitions
        # Example:
        # if is_valid = [[True, False, False], [True, False, True]],
        # is_valid has shape (buffer_size=2, n_envs=3)
        # then valid_indices = [0, 3, 5]
        # they correspond to is_valid[0, 0], is_valid[1, 0] and is_valid[1, 2]
        # or in numpy format ([rows], [columns]): (array([0, 1, 1]), array([0, 0, 2]))
        # Those indices are obtained back using np.unravel_index(valid_indices, is_valid.shape)
        valid_indices = np.flatnonzero(is_valid)
        # Sample valid transitions that will constitute the minibatch of size batch_size
        # we only return 1 episode
        #batch_size=1
        sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=True)
        # Unravel the indexes, i.e. recover the batch and env indices.
        # Example: if sampled_indices = [0, 3, 5], then batch_indices = [0, 1, 1] and env_indices = [0, 0, 2]
        batch_indices, env_indices = np.unravel_index(sampled_indices, is_valid.shape)

        # now we need to return the episodes that correspond to these indices.
        # get the episode_start and episode_lengths for these indices

        episode_starts = self.ep_start[batch_indices, env_indices]  # this should return a batch_size array of start pos
        episode_lengths = self.ep_length[batch_indices, env_indices] # this should return a batch_size array of lengths
        episode_ends = episode_starts + episode_lengths

        # now we want to return N-10 steps from the batch index
        seq_start = np.array([(batch_indices - 10) % self.buffer_size, env_indices])

        # ensure seq_starts isn't before episodes start
        seq_starts = np.maximum(seq_start,episodes_starts) # this should return shapt [batch_indices, env_indices]

        # debug
        print(seq_starts)
        print(episode_starts)
        #print(episode_lengths)
        #print(episode_ends)

        # need to generate a List of ReplayBufferSamples (tuple of tensors)
        replay_buffer_sequence_samples = []

        for ep in range(batch_size):

           # the following returns the entire episode
           #sample_idxs = np.arange(episode_starts[ep],episode_ends[ep]) % self.buffer_size
           #
           # now we only want to return N-10 steps in the episode from the select index
           sample_idxs = np.arange(seq_starts[ep],batch_indices[ep]) % self.buffer_size

           if self.optimize_memory_usage:
               next_obs = self._normalize_obs(
                   self.observations[(np.array(sample_idxs) + 1) % self.buffer_size, env_indices[ep], :], env
               )
           else:
               next_obs = self._normalize_obs(self.next_observations[sample_idxs, env_indices[ep], :], env)

           batch = (
               self._normalize_obs(self.observations[sample_idxs, env_indices[ep], :], env),
               self.actions[sample_idxs, env_indices[ep], :],
               next_obs,
               self.dones[sample_idxs],
               self.rewards[sample_idxs],
           )

           replay_buffer_sequence_samples.append(ReplayBufferSamples(*tuple(map(self.to_torch, batch))))  # type: ignore

        return replay_buffer_sequence_samples

    #def prioritised_sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
    def prioritised_sample(self, batch_size: int = 32, seq: int = 10, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        """
        Sample episode transitions from the replay buffer.

        :param batch_size: Number of sequences to sample.
        :param seq_size: Number of transitions to return in the sequence per batch
        :param env: Associated VecEnv to normalize the observations/rewards when sampling
        :return: Samples
        """
        sampled_indices, tree_idxs = [], []
        priorities = th.empty(batch_size, 1, dtype=th.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree.
        segment = self.tree.p_total / batch_size
        for i in range(batch_size):
            # extremes of the current segment
            a, b = segment * i, segment * (i + 1)

            # uniformly sample a value from the current segment
            cumsum = random.uniform(a, b)

            # tree_idx is a index of a sample in the tree, needed further to update priorities
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sampled_indices.append(int(sample_idx.item()))

        print(sampled_indices)
        # NOTE: The following has not been properly implemented upstream
        # we should be returning the weights with the samples so that the priorities
        # of those transitions once replayed can be updated in the tree
        #
        # probability of sampling transition i as P(i) = p_i^alpha / \sum_{k} p_k^alpha
        # where p_i > 0 is the priority of transition i.
        probs = priorities / self.tree.p_total

        # Importance sampling weights.
        # All weights w_i were scaled so that max_i w_i = 1.
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        # not sure why we choose a random env_index?
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(sampled_indices),))

        # now we need to return the episodes that correspond to these indices.
        # get the episode_start and episode_lengths for these indices
        # we need to return previous seq_size indices to the selected sampled_indices
        episode_starts = self.ep_end[sampled_indices, env_indices]  # this should return a batch_size array of end positions
        episode_lengths = self.ep_length[sampled_indices, env_indices] # this should return a batch_size array of lengths
        episode_ends = episode_starts + episode_lengths

        # debug
        #print(episode_starts)
        #print(episode_lengths)
        #print(episode_ends)

        # need to generate a List of ReplayBufferSamples (tuple of tensors)
        replay_buffer_sequence_samples = []
        for ep in range(batch_size):

           sample_idxs = np.arange(episode_starts[ep],episode_ends[ep]) % self.buffer_size

           if self.optimize_memory_usage:
               next_obs = self._normalize_obs(
                   self.observations[(np.array(sample_idxs) + 1) % self.buffer_size, 0, :], env
               )
           else:
               next_obs = self._normalize_obs(self.next_observations[sample_idxs, 0, :], env)

           # The following (idx,0,:) notation is ours.  quick fix as we only have 1 n_env.
           batch = (
               self._normalize_obs(self.observations[sample_idxs, 0, :], env),
               self.actions[sample_idxs, 0, :],
               next_obs,
               self.dones[sample_idxs],
               self.rewards[sample_idxs],
           )

           replay_buffer_sequence_samples.append(ReplayBufferSamples(*tuple(map(self.to_torch, batch))))  # type: ignore

        return replay_buffer_sequence_samples

