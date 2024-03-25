import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch as th
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from gymnasium import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy, MultiInputPolicy

#from stable_baselines3.common.base_class import BaseAlgorithm
#from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
#from stable_baselines3.common.policies import BasePolicy
#from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv


from sb3_contrib.drqn.policies import DRQNPolicy

SelfDRQN = TypeVar("SelfDRQN", bound="DRQN")

class DeepRecurrentQNetwork(DQN):
    """
    This implements the Deep Recurrent Q Network (paper reference)

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DRQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.0001,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

        self._last_lstm_states = None

        #print(type(self.policy)) # DRQNPolicy
        #print(type(self.q_net)) # DRQNetwork
        #print(type(self.q_net_target)) #DRQNetwork

    def _setup_model(self) -> None:
        super()._setup_model()
        
        # NOTE: implement for 1 env at the moment
        #self.single_hidden_state_shape = (self.policy.lstm_num_layers, self.n_envs, self.policy.lstm_hidden_size)
        self.single_hidden_state_shape = (self.policy.lstm_num_layers, self.policy.lstm_hidden_size)
        self._last_lstm_states = (
                th.zeros(self.single_hidden_state_shape, device=self.device),
                th.zeros(self.single_hidden_state_shape, device=self.device),
            )

    def learn(
        self: SelfDRQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DRQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDRQN:
        return super().learn(
            total_timesteps,
            callback,
            log_interval,
            tb_log_name,
            reset_num_timesteps,
            progress_bar,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        need to override DQN method to handle sequential buffer replays
        """

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        batch_size = 1
        for _ in range(gradient_steps):
                # Sample replay buffer
                # return sequence lengths within the 1-d replay data vector
                #replay_buffer_samples class is a tuple of tensors
                # for batched input it has the shape [ batch, input, envs]
                # where input is a sequence of transitions of variable lengths
                # given by the replay_data.batch_lengths index
                #
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

                #print("train")
                print("sampled replay sequence lengths {}".format(replay_data.lengths))
                # what is the effect of th.no_grad if the sequence needs to keep gradient?
                with th.no_grad():

                    # convert padded tensors to padded sequence
                    next_observations = pack_padded_sequence(replay_data.next_observations, replay_data.lengths, batch_first=True)

                    # Compute the next Q-values using the target network
                    # DQN uses these as a batch input of independent transitions
                    # We need to use it as one input sequence of transitions.
                    #print(len(replay_data.next_observations))
                    print("forward pass of q_net_target with None as h0,c0")
                    #print(next_observations.data)
                    next_q_values, _ = self.q_net_target(next_observations)
                    #print("next q values shape {}".format(next_q_values.shape)) # this should now be (batch, actions)
                    # Follow greedy policy: use the one with the highest value
                    next_q_values, _ = next_q_values.max(dim=2)
                    #print("next max q values shape {}".format(next_q_values.shape))

                    # Avoid potential broadcast issue
                    next_q_values = next_q_values.reshape(-1, 1)
                    #print("reshaped next q values {}".format(next_q_values.shape))
                    # 1-step TD target
                    # take the reward and dones for the last transition in the sequence
                    target_q_values = replay_data.rewards[:,-1] + (1 - replay_data.dones[:,-1]) * self.gamma * next_q_values

                observations = pack_padded_sequence(replay_data.observations, replay_data.lengths, batch_first=True)
                # Get current Q-values estimates
                #print(len(replay_data.observations))
                print("forward pass of q_net with None as h0,c0")
                current_q_values, _ = self.q_net(observations)
                #print("current q values shape: {}".format(current_q_values.shape)) # this should be (batch, actions) shape

                # Retrieve the q-values for the actions from the replay buffer
                # we only need to get the action for the last transition in the sequence
                #print("replay data actions shape: {}".format(replay_data.actions.shape))

                #print("replay data actions")
                #print(replay_data.actions)

                #print(replay_data.actions.reshape(1,-1))

                # reshape
                current_q_values = current_q_values.reshape(1,-1)
                actions = replay_data.actions[:,-1]
                #print("reshaped current_q_values {}".format(current_q_values.shape))
                #print("replay actions shape {}".format(replay_data.actions.shape))
                #print("reshaped actions {}".format(actions.shape))

                current_q_values = th.gather(current_q_values, dim=1, index=actions.long())

                #print("gathered current q values shape {}".format(current_q_values.shape))
                #print(target_q_values.shape)

                # Compute Huber loss (less sensitive to outliers)
                # default is to calculate mean loss across elements
                # sum doesn't seem to optimise
                # does the execution of the following create another grad_fn enabled tensor
                # that is used for backprop??
                # how is a non-reduced loss vector different to a single mean vector element
                # when it comes to backprop??
                # perhaps use with th.no_grad() when printing the following to avoid backprop interaction
                #print(F.smooth_l1_loss(current_q_values, target_q_values, reduction="none"))
                loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction="mean")
                print("loss")
                print(loss)
                losses.append(loss.item())

                # Optimize the policy
                self.policy.optimizer.zero_grad()
                loss.backward()
                # only current_q_values has the grad_fn parameters set.
                # so backprop will only apply to q_net (and not the target_q_net)

                # weights before optimisation
                #print(self.q_net.state_dict())
                # Clip gradient norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                # weights after optimisation
                #print(self.q_net.state_dict())

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    # Taken from the off_policy_algorithm class
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()

        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            # actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            # EDIT: add parameter to identify computed or random action
            # sample action get its obs from self._last_obs and (now) its lstm_state from self._last_lstm_states
            actions, buffer_actions, lstm_states, computed_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            # at the moment, we don't store lstm state in the buffer as we replay entire episode
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)
            #self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos, lstm_states)
            

            self._last_obs = new_obs
            self._last_lstm_states = deepcopy(lstm_states)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

                    # return lstm state
                    print("reset self._last_lstm_states")
                    self._last_lstm_states = (
                      th.zeros(self.single_hidden_state_shape, device=self.device),
                      th.zeros(self.single_hidden_state_shape, device=self.device),
                   )
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # we need to record hidden history of each action, regardless if random or predicted,
        # so that at prediction time, the network will be able to utilise this "history" to 
        # better predict the action to be taken at that time
        unscaled_action, lstm_states, computed_action = self.predict(self._last_obs, self._last_lstm_states, deterministic=False)
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
            computed_action = False
            # not sure if we need the following a the lstm_state won't have changed if we haven't made a forward pass

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action, lstm_states, computed_action

    ## No need to override parent classes with respect to saving to replay buffer 
    def _store_transition_disabled(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        lstm_states: Tuple[np.ndarray]
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        :param lstm_states: Tuple of the last hidden/cell state of the LSTM layer
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
            lstm_states
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the dqn predict function to include epsilon-greedy exploration.
        However, as this is a recurrent network, we need to do a forward pass for
        every step to keep the lstm hidden state for the episode.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        computed=True
        action, state = self.policy.predict(observation, state, episode_start, deterministic)
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
            computed=False
        # add computed flag
        return action, state, computed

