import torch
import torch.nn as nn

class SimpleMLPClassifier(nn.Module):
    """Simple MLP classifier with a single hidden layer."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        # Input x shape: [batch, time_frames, input_dim]
        # We might classify based on the last time step or average/pool over time
        # For now, let's process each time step independently
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        # Output shape: [batch, time_frames, num_classes] (logits)
        return x


class DeepMLPClassifier(nn.Module):
    """Deep MLP classifier with multiple hidden layers."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = [256, 128, 64], dropout: float = 0.2):
        super().__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        # Use LayerNorm instead of BatchNorm to avoid batch size issues
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Store dimensions for reference
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # For compatibility with PhonologicalLoopClassifier
        self.layer_1 = layers[0]  # First linear layer
    
    def forward(self, x: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        # Handle BatchNorm1d for different input shapes
        original_shape = x.shape
        
        # If input has more than 2 dimensions, reshape for BatchNorm1d
        if len(original_shape) > 2:
            # Flatten all dimensions except the last
            x = x.reshape(-1, original_shape[-1])
        
        # Pass through the model
        x = self.model(x)
        
        # Reshape back to original dimensions if needed
        if len(original_shape) > 2:
            new_shape = list(original_shape[:-1]) + [self.num_classes]
            x = x.reshape(new_shape)
        
        return x


class EnhancedClassifier(nn.Module):
    """
    Enhanced classifier with better capacity for S4 outputs.
    Recommended for use with the S4 layer to improve classification accuracy.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_classes)
        )
        
        # For compatibility with PhonologicalLoopClassifier
        self.layer_1 = self.model[0]  # First linear layer
    
    def forward(self, x: tensor.convert_to_tensor) -> tensor.convert_to_tensor:
        return self.model(x)
