# This will subclass the Prioritised Experience Replay
# and incorporates some of the functionality with HER
# to record episodic information within the buffer
# This currently only does non-prioritised samples
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch as th
from torch.nn.utils.rnn import pad_sequence
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.prioritized_replay_buffer import SumTree
from sb3_contrib.per.replay_partial_sequence_buffer import ReplayPartialSequenceBuffer

class PrioritizedReplayBufferSequenceSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    lstm_states: th.Tensor
    weights: Union[th.Tensor, float] = 1.0
    lengths: List = None # needed for padded sequences.
    leaf_nodes_indices: Optional[np.ndarray] = None

class PrioritizedReplaySequenceBuffer(ReplayPartialSequenceBuffer):
    """
    This provides a Replay Buffer that returns sequences of prioritized transitions.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lstm_num_layers = 1,
        lstm_hidden_size = None,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        final_beta: float = 1.0,
        min_priority: float = 1e-8

    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, lstm_num_layers=lstm_num_layers)

        assert optimize_memory_usage is False, "PrioritizedReplayBuffer does not support optimize_memory_usage=True"

        self.min_priority = min_priority
        self.alpha = alpha  # determines how much prioritization is used, alpha = 0 corresponding to the uniform case
        self.max_priority = self.min_priority  # priority for new samples, init as eps
        # Track the training progress remaining (from 1 to 0)
        # this is used to update beta
        self._current_progress_remaining = 1.0
        self.inital_beta = beta
        self.final_beta = final_beta
        self.beta_schedule = get_linear_fn(
            self.inital_beta,
            self.final_beta,
            end_fraction=1.0,
        )

        # SumTree: data structure to store priorities
        self.tree = SumTree(buffer_size=buffer_size)

    @property
    def beta(self) -> float:
        # Linear schedule
        return self.beta_schedule(self._current_progress_remaining)

    # need to override parent
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        lstm_states: Tuple[np.ndarray] = None # 2d array. for the moment whilst we replay whole episodes. othwerwise Tuple[np.ndarray]
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
        self.tree.add(self.max_priority, self.pos)

        super().add(obs, next_obs, action, reward, done, infos, lstm_states)

    # the following is an overridden version of the inherited add() method
    def _add_disabled(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        lstm_states: Tuple[np.ndarray]
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
        lstm_states_arr = np.array([[lstm_states[0].reshape(-1), lstm_states[1].reshape(-1)]])
        self.lstm_states[self.pos] = np.array(lstm_states_arr).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    # adapted from HER
    def sample(self, batch_size: int, n_prev_seq: int = 10, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample. Not used. We only return 1 episode.
        :param env: Associated VecEnv to normalize the observations/rewards when sampling
        :return: Samples
        """

        leaf_nodes_indices = np.zeros(batch_size, dtype=np.uint32)
        priorities = np.zeros((batch_size, 1))
        sample_indices = np.zeros(batch_size, dtype=np.uint32)

        # To sample a minibatch of size k, the range [0, total_sum] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree.
        segment_size = self.tree.total_sum/ batch_size

        for batch_idx in range(batch_size):
            # extremes of the current segment
            start, end = segment_size * batch_idx, segment_size * (batch_idx + 1)

            # uniformely sample a value from the current segment
            cumulative_sum = np.random.uniform(start, end)

            # leaf_node_idx is a index of a sample in the tree, needed further to update priorities
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            leaf_node_idx, priority, sample_idx = self.tree.get(cumulative_sum)

            leaf_nodes_indices[batch_idx] = leaf_node_idx
            priorities[batch_idx] = priority
            sample_indices[batch_idx] = sample_idx

        #print(sample_indices)
        # probability of sampling transition i as P(i) = p_i^alpha / \sum_{k} p_k^alpha
        # where p_i > 0 is the priority of transition i.
        probs = priorities / self.tree.total_sum

        # Importance sampling weights.
        # All weights w_i were scaled so that max_i w_i = 1.
        weights = (self.size() * probs) ** -self.beta
        weights = weights / weights.max()

        # TODO: add proper support for multi env
        # env_indices = np.random.randint(0, high=self.n_envs, size=(len(sample_idxs),))
        env_indices = np.zeros(batch_size, dtype=np.uint32)
        batch_indices = sample_indices

        print("prioritized sample indices")
        print(batch_indices)

        episode_starts = self.ep_start[batch_indices, env_indices]  # this should return a batch_size array of start pos

        # now we want to return N steps from the batch index
        seq_start = batch_indices - n_prev_seq

        # ensure seq_start isn't before episode start.
        seq_starts = np.maximum(seq_start,episode_starts)

        sample_idxs = []
        lengths = []

        # first loop is to retrieve sequences and determine max sequence length
        for ep in range(batch_size):

           seq_idxs = np.arange(seq_starts[ep],batch_indices[ep]) % self.buffer_size
           
           # edge case if sample index 0 is selected with episode starting at same position
           if len(seq_idxs) == 0:
               seq_idxs = np.array([batch_indices[ep]])

           sample_idxs.append(seq_idxs)
           lengths.append(len(seq_idxs))

        # convert list of numpys to multi-dim numpy
        # this should be easier with a known max length of N at the start
        max_length = max(lengths)

        batch_observations = np.zeros([batch_size, max_length, *self.obs_shape],dtype=self.observations.dtype)
        batch_actions = np.zeros([batch_size, max_length, self.action_dim],dtype=self.actions.dtype)
        batch_next_obs = np.zeros([batch_size, max_length, *self.obs_shape],dtype=self.next_observations.dtype)
        batch_dones = np.zeros([batch_size, max_length], dtype=self.dones.dtype)
        batch_rewards = np.zeros([batch_size, max_length], dtype=self.dones.dtype)
        batch_lstm_states = np.zeros([batch_size, max_length, *self.lstm_shape],dtype=self.lstm_states.dtype)

        for b in range(batch_size):

          env_idxs = np.full(len(sample_idxs[b]), env_indices[b])
          if self.optimize_memory_usage:
               batch_next_obs[b][0:len(sample_idxs[b])] = self._normalize_obs(
                   self.observations[(np.array(sample_idxs[b]) + 1) % self.buffer_size, env_idxs, :], env
               )
          else:
              batch_next_obs[b][0:len(sample_idxs[b])] = self._normalize_obs(self.next_observations[sample_idxs[b], env_idxs, :], env)

          batch_observations[b][0:len(sample_idxs[b])] = self._normalize_obs(self.observations[sample_idxs[b], env_idxs, : ], env)
          batch_actions[b][0:len(sample_idxs[b])] = self.actions[sample_idxs[b], env_idxs, :]
          batch_dones[b][0:len(sample_idxs[b])] = self.dones[sample_idxs[b], env_idxs]
          batch_rewards[b][0:len(sample_idxs[b])] = self.rewards[sample_idxs[b], env_idxs]
          # we really should only return the initial transition lstm state
          batch_lstm_states[b][0:len(sample_idxs[b])] = self.lstm_states[sample_idxs[b], env_idxs, :] 

        batch = (
               batch_observations,
               batch_actions,
               batch_next_obs,
               batch_dones,
               batch_rewards,
               batch_lstm_states,
               weights
        )

        #batch_lengths = np.array(lengths,dtype=np.float)

        return PrioritizedReplayBufferSequenceSamples(*tuple(map(self.to_torch, batch)),lengths,leaf_nodes_indices)  # type: ignore

    def update_priorities(self, leaf_nodes_indices: np.ndarray, td_errors: th.Tensor, progress_remaining: float) -> None:
        """
        Update transition priorities.
        :param leaf_nodes_indices: Indices for the leaf nodes to update
            (correponding to the transitions)
        :param td_errors: New priorities, td error in the case of
            proportional prioritized replay buffer.
        :param progress_remaining: Current progress remaining (starts from 1 and ends to 0)
            to linearly anneal beta from its start value to 1.0 at the end of training
        """
        # Update beta schedule
        self._current_progress_remaining = progress_remaining
        td_errors = td_errors.detach().cpu().numpy().flatten()

        for leaf_node_idx, td_error in zip(leaf_nodes_indices, td_errors):
            # Proportional prioritization priority = (abs(td_error) + eps) ^ alpha
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (abs(td_error) + self.min_priority) ** self.alpha
            self.tree.update(leaf_node_idx, priority)
            # Update max priority for new samples
            self.max_priority = max(self.max_priority, priority)

