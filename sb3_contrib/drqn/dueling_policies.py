from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor

from sb3_contrib.drqn.policies import DRQNetwork, DRQNPolicy

class DuelingDRQNModule(nn.Module):

    """
    The DQRN sequential submodule with Dueling branching

    We can't use nn.Sequential as we can't feed whole output of the LSTM
    module into the Linear module, so we need to do this

    """

    def __init__(self, features_dim, output_dim, lstm_net_arch, activation_fn: nn.Module) -> None:
        super().__init__()
        # this architecture requires an LSTM module to generate the history of obserations
        # then an MLP to approximate the Q values using the LSTM output as the input
        # 
        # we have removed the MLP at the end and just replaced it with a linear output layer.

        lstm_hidden_size = features_dim
        lstm_num_layers = len(lstm_net_arch)
        self.lstm_layers = nn.LSTM(features_dim, lstm_hidden_size, lstm_num_layers, batch_first=True)
        # the following is a bare linear output layer (no activation_fn)
        # this seems to match the description of DRQN
        # ie the first linear layer of DQN is replaced by a LSTM
        # the first linear layer in DQN is followed by an activation fn
        # not sure if this activation function is kept DRQN
        # perhaps try without and then try with.
        # the second linear layer is simply an output layer
        self.activation_fn = activation_fn
        self.val_linear_layers = nn.Sequential(
               activation_fn(),
               nn.Linear(lstm_hidden_size, 1)
        )
        self.adv_linear_layers = nn.Sequential(
               activation_fn(),
               nn.Linear(lstm_hidden_size, output_dim)
        )

    def forward(self, features: th.Tensor, lstm_states: Tuple[th.Tensor]) -> th.Tensor:

        # packed sequence will be in correct shape
        if not isinstance(features, th.nn.utils.rnn.PackedSequence):
            if features.dim() == 2:
                # batch of 1
                features = features.reshape(1,*features.shape)

        if lstm_states is not None:
          #print("lstm_states h0 shape: {}".format(lstm_states[0].shape))
          d_dim,b_dim,h_dim = lstm_states[0].shape
          if d_dim != self.lstm_layers.num_layers:
            # wrong shape. This happens with the initial zero h0,c0.
            h0 = lstm_states[0].reshape(b_dim,d_dim,h_dim)
            c0 = lstm_states[1].reshape(b_dim,d_dim,h_dim)
          else:
            h0 = lstm_states[0]
            c0 = lstm_states[1]
          lstm_states = (h0,c0)

        lstm_out, lstm_hidden_state = self.lstm_layers(features, lstm_states)

        if lstm_hidden_state[0].dim() > 2:
              # we should use the last layers h0
              linear_in = lstm_hidden_state[0][-1]
        else:
            linear_in = lstm_hidden_state[0]

        # pass latent hidden layer from lstm through the dueling linear branches
        values = self.val_linear_layers(linear_in)
        advantages = self.adv_linear_layers(linear_in)
        output = values + (advantages - advantages.mean())
        return output, lstm_hidden_state

class DuelingDRQNetwork(DRQNetwork):
    """
    Dueling Q-Network.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
            normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        action_dim = self.action_space.n  # number of actions
        self.q_net = DuelingDRQNModule(self.features_dim, action_dim, self.net_arch, self.activation_fn)

class DuelingDRQNPolicy(DRQNPolicy):
    """
    Policy class for Dueling DRQN.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def make_q_net(self) -> DuelingDRQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingDRQNetwork(**net_args).to(self.device)


