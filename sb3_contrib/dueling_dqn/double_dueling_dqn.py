from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import torch as th
import numpy as np
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.dqn.dqn import DQN

from sb3_contrib.dueling_dqn.policies import CnnPolicy, DuelingDQNPolicy, MlpPolicy, MultiInputPolicy
from sb3_contrib.dueling_dqn.dueling_dqn import DuelingDQN

class DoubleDuelingDQN(DuelingDQN):

    """
    Double Dueling Deep Q-Network (Dueling DQN)

    # this subclasses the Dueling DQN class
    # and overrides the train method to use Double DQN 
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Decouple action selection from value estimation
                # Compute q-values for the next observation using the online q net
                next_q_values_online = self.q_net(replay_data.next_observations)
                # Select action with online network
                _, next_actions_online = next_q_values_online.max(dim=1)
                # Estimate the q-values for the selected actions using target q network
                next_actions_online = next_actions_online.reshape(-1, 1) 
                next_q_values = th.gather(next_q_values, dim=1, index=next_actions_online)
                # Do we need this, given that the next_actions_online has been reshaped?
                # Avoid potential broadcast issue
                #next_q_values = next_q_values.reshape(-1, 1)

                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()


        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

