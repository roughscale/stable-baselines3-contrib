"""
Utility functions for managing LSTM states in DRQN.

This module provides helpers to standardize LSTM state shapes throughout the codebase.
PyTorch LSTM expects states in format: (num_layers, batch_size, hidden_size)
"""

from typing import Tuple
import numpy as np
import torch as th


class LSTMStateManager:
    """
    Helper class to manage LSTM state conversions and ensure consistent shapes.

    Standard PyTorch LSTM state format:
    - h_state shape: (num_layers, batch_size, hidden_size)
    - c_state shape: (num_layers, batch_size, hidden_size)

    :param num_layers: Number of LSTM layers
    :param hidden_size: LSTM hidden dimension size
    """

    def __init__(self, num_layers: int, hidden_size: int):
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def create_zero_states(
        self,
        batch_size: int,
        device: th.device
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Create zero-initialized LSTM states in standard PyTorch format.

        :param batch_size: Batch size
        :param device: PyTorch device
        :return: Tuple of (h_state, c_state), each with shape (num_layers, batch_size, hidden_size)
        """
        h_state = th.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device
        )
        c_state = th.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device
        )
        return (h_state, c_state)

    def validate_and_fix_shape(
        self,
        lstm_states: Tuple[th.Tensor, th.Tensor]
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Validate LSTM state shapes and fix if needed.

        Handles common shape inconsistencies:
        - Swapped dimensions (batch_size, num_layers, hidden_size) → (num_layers, batch_size, hidden_size)
        - Missing batch dimension

        :param lstm_states: Tuple of (h_state, c_state) tensors
        :return: Tuple of (h_state, c_state) in correct shape (num_layers, batch_size, hidden_size)
        """
        h_state, c_state = lstm_states

        # Expected shape: (num_layers, batch_size, hidden_size)
        if h_state.dim() != 3:
            raise ValueError(
                f"Expected 3D LSTM states, got shape {h_state.shape}. "
                f"LSTM states should be (num_layers, batch_size, hidden_size)"
            )

        dim0, dim1, dim2 = h_state.shape

        # Check if shape matches expected (num_layers, batch_size, hidden_size)
        if dim0 == self.num_layers and dim2 == self.hidden_size:
            # Correct shape: (num_layers, batch_size, hidden_size)
            return (h_state, c_state)

        # Check if dimensions are swapped: (batch_size, num_layers, hidden_size)
        elif dim1 == self.num_layers and dim2 == self.hidden_size:
            # Reshape to correct format
            h_fixed = h_state.transpose(0, 1).contiguous()
            c_fixed = c_state.transpose(0, 1).contiguous()
            return (h_fixed, c_fixed)

        # Check another common swap: (num_layers, hidden_size, batch_size)
        elif dim0 == self.num_layers and dim1 == self.hidden_size:
            h_fixed = h_state.transpose(1, 2).contiguous()
            c_fixed = c_state.transpose(1, 2).contiguous()
            return (h_fixed, c_fixed)

        else:
            raise ValueError(
                f"Cannot determine correct LSTM state shape from {h_state.shape}. "
                f"Expected to find num_layers={self.num_layers} and "
                f"hidden_size={self.hidden_size} in the dimensions."
            )

    def numpy_to_torch(
        self,
        numpy_states: Tuple[np.ndarray, np.ndarray],
        batch_size: int,
        device: th.device
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Convert numpy LSTM states to PyTorch tensors in correct format.

        Handles stored states from replay buffer which may be in various formats.

        :param numpy_states: Tuple of (h_state, c_state) numpy arrays
        :param batch_size: Target batch size
        :param device: PyTorch device
        :return: Tuple of (h_state, c_state) tensors in shape (num_layers, batch_size, hidden_size)
        """
        h_np, c_np = numpy_states

        # Convert to torch
        h_tensor = th.from_numpy(h_np).to(device).float()
        c_tensor = th.from_numpy(c_np).to(device).float()

        # If states are for single env, need to expand to batch
        # Expected: (num_layers, hidden_size) → (num_layers, batch_size, hidden_size)
        if h_tensor.dim() == 2:
            h_tensor = h_tensor.unsqueeze(1).expand(-1, batch_size, -1)
            c_tensor = c_tensor.unsqueeze(1).expand(-1, batch_size, -1)

        # Validate and fix shape
        return self.validate_and_fix_shape((h_tensor, c_tensor))

    def torch_to_numpy(
        self,
        torch_states: Tuple[th.Tensor, th.Tensor],
        env_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert PyTorch LSTM states to numpy for storage.

        Extracts state for a specific environment (default first env).

        :param torch_states: Tuple of (h_state, c_state) tensors
        :param env_idx: Environment index to extract (for multi-env support)
        :return: Tuple of (h_state, c_state) numpy arrays with shape (num_layers, hidden_size)
        """
        h_tensor, c_tensor = torch_states

        # Extract state for specific env
        # Input: (num_layers, batch_size, hidden_size)
        # Output: (num_layers, hidden_size)
        h_np = h_tensor[:, env_idx, :].cpu().detach().numpy()
        c_np = c_tensor[:, env_idx, :].cpu().detach().numpy()

        return (h_np, c_np)


def create_lstm_state_manager(num_layers: int, hidden_size: int) -> LSTMStateManager:
    """
    Factory function to create an LSTMStateManager.

    :param num_layers: Number of LSTM layers
    :param hidden_size: LSTM hidden dimension size
    :return: LSTMStateManager instance
    """
    return LSTMStateManager(num_layers, hidden_size)
