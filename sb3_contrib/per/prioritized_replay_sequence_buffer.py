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
from stable_baselines3.per.replaypartialsequencebuffer import ReplayPartialSequenceBuffer

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
        lstm_hidden_size = None,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage)

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

        super().__init__(obs, next_obs, action, reward, done, infos, lstm_states)

    # the following is an overridden version of the inherited add() method
    def _add(
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
        #batch_size=32
        sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=True)
        # Unravel the indexes, i.e. recover the batch and env indices.
        # Example: if sampled_indices = [0, 3, 5], then batch_indices = [0, 1, 1] and env_indices = [0, 0, 2]
        batch_indices, env_indices = np.unravel_index(sampled_indices, is_valid.shape)

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
               batch_lstm_states
        )

        #batch_lengths = np.array(lengths,dtype=np.float)

        return ReplaySequenceBufferSamples(*tuple(map(self.to_torch, batch)),lengths)  # type: ignore
