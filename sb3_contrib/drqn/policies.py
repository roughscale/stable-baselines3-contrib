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

# subclass torch layers to inspect internals
# may need to implement super to inherit. what are the init params?
class DRQNLSTM(nn.LSTM):
    def __init__(
           input_size,
           hidden_size,
           num_layers = 1,
           bias = True,
           batch_first = False,
           dropout = 0,
           bidirectional = False,
           proj_size = 0
    ):
        super().__init__(
                input_size = input_size,
                hidden_size = hidden_size,
                num_layers = num_layers,
                bias = bias,
                batch_first = batch_first,
                dropout = dropout,
                bidirectional = bidirectional,
                proj_size = proj_size
                )

class DRQNLinear(nn.LSTM):
    def __init__(
        in_features,
        out_features,
        bias = True
        ):

        super().__init__(
            in_features = in_features,
            out_features = out_features,
            bias = bias
            )

class DRQNModule(nn.Module):

    """
    The DQRN sequential submodule

    We can't use nn.Sequential as we can't feed whole output of the LSTM
    module into the Linear module, so we need to do this

    :param lstm_layers: LSTM module
    :param linear_layer: Linear module
    """

    def __init__(self, lstm_layers: nn.Module, activation_fn: nn.Module, linear_layers: nn.Sequential) -> None:
        super().__init__()
        # this architecture requires an LSTM module to generate the history of obserations
        # then an MLP to approximate the Q values using the LSTM output as the input
        self.lstm_layers = lstm_layers
        self.activation_fn = activation_fn
        self.linear_layers = linear_layers

    def forward(self, features: th.Tensor, lstm_states: Tuple[th.Tensor]) -> th.Tensor:
        #print("DRQNModule forward")
        #print("input shape")
        #if isinstance(features, th.nn.utils.rnn.PackedSequence):
        #    print(features.data.shape)
        #else:
        #    print(features.shape)
        # input could be non-sequential
        # of shape [ n_envs, input_dim ]
        # batched sequential input is [N, L, H] 
        # [ batch_size, seq_len, input_dim ]
        # otherwise unbatched sequential input 
        # is [seq_len, input_dim]
        # how to distinguish between non-sequential input
        # and unbatched sequential input?
        # perhaps don't need to distinguish for the moment
        # non-seq input is passed as [1, H] 
        # so it is effectively a sequence of 1
        # this assumption won't hold for multi-env input

        #if lstm_states is not None:
        #    print("input lstm states")
        #    print(type(lstm_states))
        #    print(len(lstm_states))
        #    print(lstm_states[0].shape)
        
        lstm_out, lstm_hidden_state = self.lstm_layers(features, lstm_states)
        # for batched input lstm_out would be [N, L, out_dim]
        # otherwise [L, out_dim]
        # 
        # and lstm_hidden_state would be [ D * num_lstm_layers, N, out_dim]
        # or just [ D * num_lstm_layers, out_dim ] for unbatched
        # where D is 1 for uni-directional, 2 for bi-directional
        #
        # however we only want to use the output of the last element 
        # which is the representation of the state at that timestep
        # 
        # the lstm_hidden_state seems to be the hidden_state
        # for the last timestep
        # (h0,c0)
        # use h0 as input for the linear layer
        # check

        #print(lstm_hidden_state[0].shape)
        #print(features.shape)
        #print(lstm_out.shape)
        #print(lstm_out.reshape(-1))

        #if features.shape[0] > 2:
        #  print(lstm_out[:,-1,:])
        #  print(th.eq(lstm_hidden_state[0], lstm_out[:,-1,:]))
        #else:
        #  print(lstm_out[-1])
        ##  print(lstm_hidden_state[0] == lstm_out[-1])
        # lstm_hidden_state is a tuple of tensors
        # get h0 part of tuple and first element for unidirectional LSTM
        #print("lstm out")
        # lstm_hidden_size[0] is [ D*num_layers, N , H ] for batched
        # or [ D*num_lstm_layers, H ] for unbatched
        if lstm_hidden_state[0].dim() > 2:
            linear_in = lstm_hidden_state[0][0]
        else:
            linear_in = lstm_hidden_state[0]
        #print(linear_in.shape)
        # the linear layer input should be [ N, H ]
        output = self.linear_layers(linear_in)
        #print("module output")
        #print(output.shape)
        # output shape is [ N , action_dim]
        return output, lstm_hidden_state

class DRQNetwork(BasePolicy):
    """
    Deep Recurrent Q Network

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        lstm_num_layers: int,
        lstm_hidden_size: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        # here net_arch refers to the lstm network
        #if net_arch is None:
        # will assume it is 1 layer of observation_space length
        action_dim = self.action_space.n  # number of actions

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        action_dim = int(self.action_space.n)  # number of actions
        # for multilayer LSTMs will need to implement something like the create_mlp function
        lstm_layers = nn.LSTM(self.features_dim, lstm_hidden_size, lstm_num_layers, batch_first=True)
        # the following is a bare linear output layer (no activation_fn)
        # this seems to match the description of DRQN
        # ie the first linear layer of DQN is replaced by a LSTM
        # the first linear layer in DQN is followed by an activation fn
        # not sure if this activation function is kept DRQN
        # perhaps try without and then try with.
        # the second linear layer is simply an output layer
        linear_layers = nn.Sequential(
               nn.Linear(lstm_hidden_size, action_dim)
        )
        # the following adds an MLP as a linear layer
        #linear_layers = nn.Sequential(
        #        nn.Linear(lstm_hidden_size,lstm_hidden_size),
        #        self.activation_fn(),
        #        nn.Linear(lstm_hidden_size,action_dim)
        #)
        self.q_net = DRQNModule(lstm_layers, activation_fn, linear_layers)

    def forward(self, obs: th.Tensor, lstm_states: th.Tensor = None) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        #print(obs)
        #print(type(obs))
        #print(self.extract_features(obs, self.features_extractor))
        # is this a batched input??
        #print("DRQN forward")
        #print(obs.shape)
        # This is a tensor of size [ep_length, feature_vector_size] which seems correct
        #features = self.extract_features(obs, self.features_extractor) # this should return the same obs
        #print(features.shape))
        # this returns the same size as obs
        #forward_result = self.q_net(features)
        #print(forward_result)
        #return forward_result
        # extract_features doesn't support packed padded sequence
        #return self.q_net(self.extract_features(obs, self.features_extractor), lstm_states)
        return self.q_net(obs, lstm_states)


    def _predict(self, observation: th.Tensor, last_lstm_states: Tuple[th.Tensor],  deterministic: bool = True) -> th.Tensor:
        # print("drqn _predict")
        # observation is in the shape [n_envs, input_dim]
        q_values, lstm_states = self(observation, last_lstm_states)
        #print("drqn_predict after")
        # Greedy action
        #print(q_values.shape)
        #print(type(lstm_states))
        #print(lstm_states[0].shape)
        # the following is from DQN.  What shape are the q-values in DQN?
        #action = q_values.argmax(dim=1).reshape(-1)
        action = q_values.argmax(dim=1).reshape(-1)
        #print(action)
        return action, lstm_states

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

class DRQNPolicy(BasePolicy):
    """
    Deep Recurrent Q-Network.

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
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images
        )


        action_dim = self.action_space.n  # number of actions

        # here net_arch refers to the lstm network
        # currently hardcoded as 1 layer of features_dim
        self.net_arch = net_arch
        self.lstm_num_layers = 1
        self.lstm_hidden_size = self.net_arch[0]
        print(net_arch)

        self.activation_fn = activation_fn

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "lstm_num_layers": self.lstm_num_layers,
            "lstm_hidden_size": self.lstm_hidden_size,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images
        }

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        #print("network parameters")
        #print(list(self.parameters()))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
            
    def make_q_net(self) -> DRQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DRQNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, lstm_states: Tuple[th.Tensor], deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, lstm_states, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs
            )
        )
        return data

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions, state = self._predict(observation, state, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

