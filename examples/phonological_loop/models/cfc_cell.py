# phonological_loop/models/cfc_cell.py

from ray import get
from ember_ml.nn.modules.activations.tanh_module import Tanh as LeCun
import numpy as np
from ember_ml.nn.tensor.types import TensorLike
from ember_ml.nn.modules import Module
from ember_ml.nn.modules.activations import get_activation
class CfCCell(Module):
    """
    Standalone PyTorch implementation of a Closed-form Continuous-time (CfC) RNN cell.
    Based on the logic from the provided ncps.torch implementation.
    Processes a single time step.
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        mode="default",
        backbone_activation="lecun_tanh",
        backbone_units=128,
        backbone_layers=1,
        backbone_dropout=0.0,
    ):
        super(CfCCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                f"Unknown mode '{mode}', valid options are {str(allowed_modes)}"
            )
        self.mode = mode
        from ember_ml.nn.modules.activations import get_activation
        # Activation mapping
        if backbone_activation == "silu":
            activation_module = get_activation("silu")
        elif backbone_activation == "relu":
            activation_module = get_activation("relu")
        elif backbone_activation == "tanh":
            activation_module = get_activation("tanh")
        elif backbone_activation == "gelu":
            activation_module = get_activation("gelu")
        elif backbone_activation == "lecun_tanh":
            activation_module = get_activation("lecun_tanh")
        else:
            raise ValueError(f"Unknown activation {backbone_activation}")

        # Backbone network (optional MLP before core CfC logic)
        self.backbone = None
        self.backbone_layers = backbone_layers
        if backbone_layers > 0:
            layer_list = [
                nn.Linear(input_size + hidden_size, backbone_units),
                activation_module(),
            ]
            for i in range(1, backbone_layers):
                layer_list.append(nn.Linear(backbone_units, backbone_units))
                layer_list.append(activation_module())
                if backbone_dropout > 0.0:
                    layer_list.append(torch.nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)

        # Core CfC layers
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # Determine input shape for ff layers based on backbone presence
        ff_input_shape = int(
            hidden_size + input_size if backbone_layers == 0 else backbone_units
        )

        self.ff1 = nn.Linear(ff_input_shape, hidden_size)
        if self.mode == "pure":
            # Parameters for the "pure" mode (direct solution approximation)
            self.w_tau = nn.Parameter(
                data=torch.zeros(1, hidden_size), requires_grad=True
            )
            self.A = nn.Parameter(
                data=torch.ones(1, hidden_size), requires_grad=True
            )
        else:
            # Parameters for "default" and "no_gate" modes
            self.ff2 = nn.Linear(ff_input_shape, hidden_size)
            self.time_a = nn.Linear(ff_input_shape, hidden_size)
            self.time_b = nn.Linear(ff_input_shape, hidden_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights using Xavier uniform distribution."""
        for name, param in self.named_parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_uniform_(param)
            elif "bias" in name and param.requires_grad:
                 nn.init.zeros_(param) # Initialize biases to zero

    def forward(self, input, hx, ts=1.0):
        """
        Processes a single time step.

        Args:
            input (tensor.convert_to_tensor): Input tensor for the current time step, shape (batch, input_size).
            hx (tensor.convert_to_tensor): Hidden state from the previous time step, shape (batch, hidden_size).
            ts (float or tensor.convert_to_tensor): Time span for the current step. Defaults to 1.0.

        Returns:
            Tuple[tensor.convert_to_tensor, tensor.convert_to_tensor]: Tuple containing the output and the new hidden state,
                                               both of shape (batch, hidden_size).
                                               In this implementation, output and hidden state are the same.
        """
        # Concatenate input and previous hidden state
        x = torch.cat([input, hx], dim=1)

        # Pass through backbone if it exists
        if self.backbone is not None:
            x = self.backbone(x)

        # Core CfC logic
        ff1_output = self.ff1(x)

        if self.mode == "pure":
            # Direct solution approximation
            # Ensure ts is compatible for broadcasting if it's a tensor
            if isinstance(ts, TensorLike) and ts.dim() == 1:
                ts = ts.unsqueeze(1) # Shape (batch, 1)

            new_hidden = (
                -self.A
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1_output)))
                * ff1_output
                + self.A
            )
        else:
            # Default or no_gate mode
            ff2_output = self.ff2(x)
            ff1_act = self.tanh(ff1_output)
            ff2_act = self.tanh(ff2_output)

            t_a_output = self.time_a(x)
            t_b_output = self.time_b(x)

            # Ensure ts is compatible for broadcasting if it's a tensor
            if ts.dim() == 1:
                ts = ts.unsqueeze(1) # Shape (batch, 1)

            t_interp = self.sigmoid(t_a_output * ts + t_b_output)

            if self.mode == "no_gate":
                new_hidden = ff1_act + t_interp * ff2_act
            else: # mode == "default"
                new_hidden = ff1_act * (1.0 - t_interp) + t_interp * ff2_act

        # In this cell, output and hidden state are the same
        output = new_hidden
        return output, new_hidden