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

class ReplaySequenceBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    lstm_states: th.Tensor
    lengths: List # needed for padded sequences.

class ReplayPartialSequenceBuffer(ReplayBuffer):
    """
    This provides a Replay Buffer that returns episodic sequences of transitions.

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

        # Episodic params from HER
        self.ep_start = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.ep_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self._current_ep_start = np.zeros(self.n_envs, dtype=np.int64)

        # LSTM state information
        # a 2-cell tuple of tensors
        # need to know shape of the lstm_state
        #print("pri seq buff init")
        #print(lstm_hidden_size)
        # disable lstm_hidden_size for now
        # lets assume lstm_hidden_state is the same as obs_shape
        lstm_hidden_size = self.obs_shape[0]
        print(self.obs_shape)
        print(lstm_hidden_size)
        # lstm state outputs from pytorch is normally a 2 element tuple of tensors (h0,c0)
        # we convert this to 2-dim ndarray
        self.lstm_shape = np.array((2,lstm_hidden_size))
        print(self.lstm_shape)
        #print(self.lstm_shape.shape)
        #if False: # disable for now
        #  self.lstm_states = (
        #             np.zeros((self.buffer_size, self.n_envs), dtype=np.float32),
        #             np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        #  )
        #else:
        self.lstm_states = np.zeros((self.buffer_size, self.n_envs, *self.lstm_shape), dtype=np.float32)
        #print("lstm_states")
        #print(self.lstm_states)
        #print("lstm_states shape")
        #print(self.lstm_states.shape)
        #self.lstm_states = (
        #            np.zeros((self.buffer_size, self.n_envs), dtype=np.float32),
        #            np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        #)

    # from PER
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
        #super().add(obs, next_obs, action, reward, done, infos)
        #print("buff add")
        # what are the dimensions of the obs
        #print(obs)
        #print(obs.shape)
        self._add(obs, next_obs, action, reward, done, infos, lstm_states)

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
        # At the moment, we don't store the lstm_states in the replay buffer
        # as we replay the entire episode
        # if we use randomised replay, we may wish to store this
        # altho the original paper zerod this state (test for performance loss in storing/retrieving this state)
        #print("buffer _add")
        # lstm_states seems to be a tuple of np.array
        #print(lstm_states)
        #print(lstm_states.shape)
        #lstm_states = lstm_states.reshape((self.n_envs, self.lstm_shape))
        #print(self.lstm_states[self.pos])
        #print(self.lstm_states[self.pos].shape)
        #self.lstm_states[self.pos] = np.array(lstm_states).copy()

        # set lstm_states to zero
        # lstm_states will be a 2 element tuple of size [1, lstm_hidden_size]
        # the 1 indicates unidirectoral LSTM but we can treat this as n_env
        # convert tuple to ndarray
        # QUICK HACK to match the shape [n_envs, 2, lstm_hidden_size ]
        #print(lstm_states[0].shape)
        #print(lstm_states[0].reshape(-1).shape)
        #print(np.array([lstm_states[0].reshape(-1),lstm_states[1].reshape(-1)]).shape)
        lstm_states_arr = np.array([[lstm_states[0].reshape(-1), lstm_states[1].reshape(-1)]])
        #print(lstm_states_arr.shape)
        # this is shape [1,2,hidden_size]
        self.lstm_states[self.pos] = np.array(lstm_states_arr).copy()
        # self.lstm_states should match shape of obs/new_obs
        # [ n_envs, lstm_shape ] 

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

        # batch_indices and env_indices are both [batch_size,] shape

        # now we need to return the episodes that correspond to these indices.
        # get the episode_start and episode_lengths for these indices

        episode_starts = self.ep_start[batch_indices, env_indices]  # this should return a batch_size array of start pos
        # This returns the partial sequence from beginning up until sampled transition.
        # We no longer return full sequence
        #episode_lengths = self.ep_length[batch_indices, env_indices] # this should return a batch_size array of lengths
        #episode_ends = episode_starts + episode_lengths

        # now we want to return N steps from the batch index
        seq_start = batch_indices - n_prev_seq

        # ensure seq_start isn't before episode start.
        seq_starts = np.maximum(seq_start,episode_starts)

        # debug
        #print(seq_starts)
        #print(seq_starts.shape)
        #print(batch_indices.shape)
        #print(episode_starts)
        #print(episode_lengths)
        #print(episode_ends)


        sample_idxs = []
        lengths = []

        # first loop is to retrieve sequences and determine max sequence length
        for ep in range(batch_size):

           #print(ep)
           # the following returns the entire episode
           #sample_idxs = np.arange(episode_starts[ep],episode_ends[ep]) % self.buffer_size
           #
           # the following returns the partial episode up until transition
           seq_idxs = np.arange(seq_starts[ep],batch_indices[ep]) % self.buffer_size
           #print(seq_idxs)
           #print("seq idxs shape")
           #print(seq_idxs.shape)
           
           # edge case if sample index 0 is selected with episode starting at same position
           if len(seq_idxs) == 0:
               seq_idxs = np.array([batch_indices[ep]])

           sample_idxs.append(seq_idxs)
           lengths.append(len(seq_idxs))


        # convert list of numpys to multi-dim numpy
        # this should be easier with a known max length of N at the start
        max_length = max(lengths)
        #print("max_length: {}".format(max_length))
        #seq_idxs = np.zeros([batch_size, max_length])d
        #for i in range(batch_size):
        #    seq_idxs[i,0:len(sample_idxs[i])] = sample_idxs[i]

        # not sure if we are not able to query a single-dim tensor using multi-dim indices and return a multi-dim tensor
        # ie. acting as a batched query of single-dim indices of the single-dim tensor
        batch_observations = np.zeros([batch_size, max_length, *self.obs_shape],dtype=self.observations.dtype)
        batch_actions = np.zeros([batch_size, max_length, self.action_dim],dtype=self.actions.dtype)
        batch_next_obs = np.zeros([batch_size, max_length, *self.obs_shape],dtype=self.next_observations.dtype)
        batch_dones = np.zeros([batch_size, max_length], dtype=self.dones.dtype)
        batch_rewards = np.zeros([batch_size, max_length], dtype=self.dones.dtype)
        batch_lstm_states = np.zeros([batch_size, max_length, *self.lstm_shape],dtype=self.lstm_states.dtype)

        # second loop is to populate multi-dim nparray (padding out sequences with zeros)

        for b in range(batch_size):

          env_idxs = np.full(len(sample_idxs[b]), env_indices[b])
          #print("env idxs")
          #print(env_idxs)
          if self.optimize_memory_usage:
               batch_next_obs[b][0:len(sample_idxs[b])] = self._normalize_obs(
                   self.observations[(np.array(sample_idxs[b]) + 1) % self.buffer_size, env_idxs, :], env
               )
          else:
              #print(sample_idxs[b])
              #print(sample_idxs[b].shape)
              #print(batch_next_obs[b])
              # self.next_obs is [ buffer_size, n_envs, obs_dim] size.
              # we need to sample a sequence of the first dimension
              # we need to broadcast the env_indices to equal the sequence len
              #print("batch next obs [b] shape")
              #print(batch_next_obs[b].shape)
              #print(self.next_observations[sample_idxs[b], env_idxs, :])
              #print("next_obs shape")
              # this provides the correct shape (seq length of obs_dim)
              #print(self.next_observations[sample_idxs[b], env_idxs, :].shape)
              # the following provides incorrect shape (it removes dim 2)
              #print(self.next_observations[sample_idxs[b],0,:].shape)
              #print(self._normalize_obs(self.next_observations[sample_idxs[b]], env).shape)
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
