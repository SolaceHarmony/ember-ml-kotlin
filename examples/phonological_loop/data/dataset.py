import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from phonological_loop.utils.audio import load_wav_to_torch

class ModulationDataset(Dataset):
    """Simple dataset for loading pre-generated noisy WAV files."""
    def __init__(self, data_dir, label_map, target_len=16000):
        self.data_dir = data_dir
        self.label_map = label_map # {classname: index}
        self.filenames = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        # Normalize all keys to lowercase for robust matching
        print(f"ModulationDataset.__init__ - label_map: {label_map}")
        if isinstance(next(iter(label_map.keys())), str):
            # If label_map is already in {name: idx} format
            self.class_to_idx = {name.lower(): idx for name, idx in label_map.items()}
        else:
            # If label_map is in {idx: name} format, invert it
            self.class_to_idx = {name.lower(): idx for idx, name in label_map.items()}
        print(f"ModulationDataset.__init__ - class_to_idx: {self.class_to_idx}")
        self.target_len = target_len

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.data_dir, filename)
        waveform, sr = load_wav_to_torch(filepath)

        # Determine label from filename (example convention: "AM_signal_01.wav")
        class_name = filename.split('_')[0]
        # Always use lowercase for consistent lookup since class_to_idx keys are normalized to lowercase
        label = self.class_to_idx.get(class_name.lower(), -1)

        if waveform is None or label == -1:
            print(f"Warning: Skipping invalid file or label for {filename}")
            # Return dummy data or skip in collate_fn
            return torch.zeros(self.target_len), tensor.convert_to_tensor(-1) # Return dummy data

        # Pad or truncate
        current_len = waveform.shape[0]
        if current_len > self.target_len:
            waveform = waveform[:self.target_len]
        elif current_len < self.target_len:
            padding = self.target_len - current_len
            waveform = F.pad(waveform, (0, padding))

        return waveform, tensor.convert_to_tensor(label, dtype=torch.long)