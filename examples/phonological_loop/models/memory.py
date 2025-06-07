import torch
import torch.nn as nn

class PhonologicalLoopMemory(nn.Module):
    """
    Phonological Loop Memory module that maintains a buffer of recent feature windows
    and implements decay and rehearsal mechanisms.

    As described in the paper:
    - Maintains a buffer of the last `buffer_len` feature windows
    - Applies exponential decay to all windows in the buffer
    - Implements a simple rehearsal mechanism (e.g., based on feature energy or a fixed pattern)
    - Composes a state for the classifier by concatenating recent windows and rehearsal state
    """
    def __init__(self,
                 feature_dim: int = 5,  # Updated default based on AnalyticSignalExtractor
                 window_len: int | None = None, # Allow lazy init
                 buffer_len: int = 10,   # Number of windows to keep in buffer
                 decay_factor: float = 0.9,  # Exponential decay factor
                 num_recent_windows: int = 3):  # Number of recent windows to include in composed state
        super().__init__()
        
        self.feature_dim = feature_dim
        self.window_len = window_len
        self.buffer_len = buffer_len
        self.decay_factor = decay_factor
        self.num_recent_windows = min(num_recent_windows, buffer_len)
        
        # Store decay factor as a registered buffer for device compatibility
        self.register_buffer('decay', tensor.convert_to_tensor(decay_factor))
        
        # Initialize buffer placeholders - actual buffers will be created in _initialize_buffers
        # Use regular attributes instead of register_buffer with None
        self.feature_buffer = None
        self.rehearsal_state = None
        self.current_pos = None
        self.buffer_filled = None
        
        # Register a dummy buffer to keep .to() path alive
        self.register_buffer('_dummy', torch.zeros(0))
        
        # Track if buffers are initialized
        self.initialized = False
    
    def _initialize_buffers(self, batch_size, device):
        """Initialize buffers for the given batch size"""
        # Create new tensors
        self.feature_buffer = torch.zeros(
            batch_size, self.buffer_len, self.feature_dim, self.window_len,
            device=device
        )
        self.rehearsal_state = torch.zeros(
            batch_size, self.feature_dim, self.window_len,
            device=device
        )
        self.current_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.buffer_filled = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        self.initialized = True
    
    def forward(self, features):
        """
        Process a new batch of features through the phonological loop.

        Args:
            features: Input features [batch, feature_dim, window_len]

        Returns:
            composed_state: Concatenated state for classifier [batch, feature_dim * (num_recent_windows + 1) * window_len]
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Initialize buffers if needed or if batch size changed
        if not self.initialized or (self.feature_buffer is not None and
                                   self.feature_buffer.size(0) != batch_size):
            self._initialize_buffers(batch_size, device)
        
        # Process each item in the batch
        composed_states = []
        # Apply decay to all buffers (vectorized across batch)
        # Use self.decay buffer instead of self.decay_factor attribute
        self.feature_buffer.mul_(self.decay)
        
        # Store old positions before updating
        old_pos = self.current_pos.clone()
        
        # Add new windows to buffer (vectorized across batch)
        batch_indices = torch.arange(batch_size, device=device)
        self.feature_buffer[batch_indices, self.current_pos] = features
        
        # Update buffer positions (vectorized)
        self.current_pos = (self.current_pos + 1) % self.buffer_len
        # Mark buffers as filled where current_pos wrapped to 0
        self.buffer_filled = torch.logical_or(self.buffer_filled, self.current_pos == 0)
        
        for b in range(batch_size):
            # 4. Rehearsal mechanism (Simplified - using most recent window as placeholder)
            # TODO: Implement a more sophisticated rehearsal trigger if needed (e.g., based on energy)
            # For now, let's just use the most recently added window as the rehearsal state
            # This mimics a simple "recency" effect.
            rehearsal_pos = old_pos[b]
            self.rehearsal_state[b] = self.feature_buffer[b, rehearsal_pos].clone()
            
            # 5. Compose state for classifier
            # Get the last num_recent_windows from buffer
            recent_windows = []
            for i in range(self.num_recent_windows):
                pos = (old_pos[b] - i) % self.buffer_len
                if pos < 0:
                    pos += self.buffer_len
                
                # Only include if buffer has been filled up to this point
                # Determine how many valid past windows exist
                num_valid_windows = self.buffer_len if self.buffer_filled[b] else old_pos[b] + 1
                if i < num_valid_windows:
                    recent_windows.append(self.feature_buffer[b, pos])
                else:
                    # Pad with zeros if buffer not filled - use features.new_zeros for device safety
                    recent_windows.append(features.new_zeros(self.feature_dim, self.window_len))
            
            # Concatenate recent windows and rehearsal state
            # [feature_dim * (num_recent_windows + 1) * window_len]
            state_components = recent_windows + [self.rehearsal_state[b]]
            composed_state = torch.cat([window.flatten() for window in state_components])
            composed_states.append(composed_state)
        
        # Stack batch dimension
        return torch.stack(composed_states)
    
    def reset(self):
        """Reset the memory state for all batch elements"""
        if self.initialized:
            self.feature_buffer.zero_()
            self.rehearsal_state.zero_()
            self.current_pos.zero_()
            self.buffer_filled.fill_(False)
    
    def reset_batch_element(self, batch_idx):
        """Reset the memory state for a specific batch element"""
        if self.initialized and batch_idx < self.feature_buffer.size(0):
            self.feature_buffer[batch_idx].zero_()
            # self.mask_buffer[batch_idx].zero_() # Removed
            self.rehearsal_state[batch_idx].zero_()
            self.current_pos[batch_idx] = 0
            self.buffer_filled[batch_idx] = False