from ember_ml import ops
from ember_ml.nn import tensor
from typing import Optional

class WaveFeatureExtractor:
    """Extract features from waveform data using backend-agnostic operations.
    
    Uses sliding windows and frequency-domain transformations while maintaining
    backend purity through the ops abstraction layer.
    """
    
    def __init__(self, window_size: int, hop_length: Optional[int] = None):
        self.window_size = window_size
        self.hop_length = hop_length or window_size // 2
        
    def extract(self, waveform: tensor.Tensor) -> tensor.Tensor:
        """Extract frequency-domain features from waveform.
        
        Args:
            waveform: Input tensor of shape (batch_size, time_steps)
            
        Returns:
            Features tensor of shape (batch_size, n_features, n_frames)
        """
        x = tensor.convert_to_tensor(waveform)
        
        # Frame the signal into overlapping windows
        frames = ops.frame(x, 
                         frame_length=self.window_size,
                         hop_length=self.hop_length)
                         
        # Apply Hann window
        window = ops.hann_window(self.window_size)
        frames = ops.multiply(frames, window)
        
        # Transform to frequency domain
        features = ops.stft(frames)
        
        return features
