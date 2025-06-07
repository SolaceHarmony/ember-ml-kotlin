import os
import torch
import torch.nn.functional as F
from scipy.io import wavfile
import importlib.util
import numpy as np

# Check if torchaudio is available
torchaudio_available = importlib.util.find_spec("torchaudio") is not None
if torchaudio_available:
    try:
        import torchaudio
        print("torchaudio loaded successfully!")
    except Exception as e:
        print(f"Error loading torchaudio: {e}")
        torchaudio_available = False

# Check if Dia TTS is available
dia_available = importlib.util.find_spec("dia") is not None
if dia_available:
    try:
        from dia.model import Dia
        
        # Initialize Dia TTS model
        try:
            # Determine device
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
                
            print(f"Loading Dia TTS model on {device}...")
            dia_model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device)
            print("Real TTS engine (Dia) loaded successfully!")
            tts_available = True
        except Exception as e:
            print(f"Error loading Dia TTS model: {e}")
            tts_available = False
    except Exception as e:
        print(f"Error importing Dia: {e}")
        dia_available = False
        tts_available = False
else:
    tts_available = False
    print("Dia TTS not found, will use placeholder TTS")

def load_wav_to_torch(wav_path, target_sr=None):
    """Load a WAV file as a torch tensor."""
    if not os.path.exists(wav_path):
        print(f"File {wav_path} does not exist")
        return None, None
    
    try:
        sr, wav = wavfile.read(wav_path)
        # Convert to float and normalize
        if wav.dtype == 'int16':
            wav = wav.astype('float32') / 32768.0
        elif wav.dtype == 'int32':
            wav = wav.astype('float32') / 2147483648.0
        elif wav.dtype == 'uint8':
            wav = (wav.astype('float32') - 128) / 128.0
        
        # Convert to torch tensor
        wav = torch.FloatTensor(wav)
        
        # Resample if needed
        if target_sr is not None and target_sr != sr:
            # This is a placeholder for resampling
            # In a real implementation, you would use torchaudio.transforms.Resample
            print(f"Warning: Resampling not implemented. Original sr={sr}, target sr={target_sr}")
        
        return wav, sr
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        return None, None

def generate_tts_speech(text, filename=None, sample_rate=16000):
    """
    Generate speech from text using a real TTS engine or fall back to a placeholder.
    
    This function attempts to use the Dia TTS model if available.
    If not available, it falls back to a simple placeholder that generates
    a sine wave with some variation based on the input text.
    
    The paper requires male voices for the simulations, so we use speaker 1 [S1]
    in the Dia model which has a male voice, as specified in the original paper:
    "Voice message signals m(t) were obtained via a 64 kHz digitisation of male speech."
    
    Args:
        text (str): Text to convert to speech
        filename (str, optional): Path to save the audio file
        sample_rate (int): Desired sample rate of the output audio
        
    Returns:
        tuple: (filename, sample_rate)
    """
    if tts_available and dia_available:
        try:
            # Use real TTS engine from Dia
            print(f"Generating real TTS for: '{text}' -> {filename}")
            
            # Format text for Dia with male voice (speaker 1)
            # ALWAYS use the male voice (S1) as required by the paper
            # Strip any existing speaker tags and add [S1]
            if text.startswith("[S1]") or text.startswith("[S2]"):
                # Remove existing speaker tag
                text = text[4:].strip()
            
            # Always add the male speaker tag
            formatted_text = f"[S1] {text}"
            print(f"Using male voice with text: '{formatted_text}'")
                
            # Generate speech with Dia
            with torch.inference_mode():
                output_audio_np = dia_model.generate(
                    formatted_text,
                    max_tokens=1024,  # Adjust based on text length
                    cfg_scale=3.0,
                    temperature=1.3,
                    top_p=0.95,
                    use_cfg_filter=True,
                    cfg_filter_top_k=30,
                    use_torch_compile=False,
                )
            
            # Dia outputs at 44.1kHz
            output_sr = 44100
            
            # Resample if needed
            if sample_rate != output_sr and torchaudio_available:
                waveform_tensor = tensor.convert_to_tensor(output_audio_np).unsqueeze(0)  # [1, samples]
                resampler = torchaudio.transforms.Resample(
                    orig_freq=output_sr,
                    new_freq=sample_rate
                )
                resampled_waveform = resampler(waveform_tensor).squeeze(0).numpy()
                output_audio_np = resampled_waveform
                output_sr = sample_rate
            
            # Save to file if filename is provided
            if filename:
                # Convert to int16 for saving
                output_audio_int16 = (output_audio_np * 32767).astype(tensor.int16)
                wavfile.write(filename, output_sr, output_audio_int16)
                print(f"Saved real TTS audio to {filename}")
            
            return filename, output_sr
                
        except Exception as e:
            print(f"Error using real TTS engine: {e}")
            print("Falling back to placeholder TTS...")
    
    # Fallback to placeholder if real TTS is not available or fails
    print(f"[Placeholder] Generating TTS for: '{text}' -> {filename}")
    
    # Generate a dummy audio signal (sine wave with some variation based on text)
    duration = 0.5 + 0.1 * len(text)  # Duration based on text length
    t = torch.linspace(0, duration, int(duration * sample_rate))
    
    # Use hash of text to create some variation in the frequency
    text_hash = sum(ord(c) for c in text)
    freq = 200 + (text_hash % 400)  # Frequency between 200-600 Hz
    
    # Generate a simple sine wave
    signal = torch.sin(2 * torch.pi * freq * t)
    
    # Add some amplitude modulation
    am = 0.5 + 0.5 * torch.sin(2 * torch.pi * 2 * t)
    signal = signal * am
    
    # Normalize
    signal = 0.9 * signal / torch.max(torch.abs(signal))
    
    # Save to file if filename is provided
    if filename:
        # Convert to int16
        signal_int = (signal * 32767).to(torch.int16)
        wavfile.write(filename, sample_rate, signal_int.numpy())
        print(f"[Placeholder] Saved dummy TTS audio to {filename}")
    
    return filename, sample_rate

# Keep the old name for backward compatibility
generate_tts_speech_placeholder = generate_tts_speech

def modulate_am_torch(message, sample_rate, carrier_freq):
    """Amplitude modulate a message signal onto a carrier using PyTorch."""
    # Create time vector
    duration = message.shape[0] / sample_rate
    t = torch.linspace(0, duration, message.shape[0], device=message.device)
    
    # Create carrier
    carrier = torch.cos(2 * torch.pi * carrier_freq * t)
    
    # Normalize message to [0, 1] range for AM
    message_norm = (message - message.min()) / (message.max() - message.min() + 1e-10)
    
    # Apply AM modulation (carrier * (1 + message))
    modulated = carrier * (1 + message_norm)
    
    # Normalize output
    modulated = 0.9 * modulated / torch.max(torch.abs(modulated))
    
    return modulated

def modulate_fm_torch(message, sample_rate, carrier_freq, modulation_index=1.0):
    """Frequency modulate a message signal onto a carrier using PyTorch."""
    # Create time vector
    duration = message.shape[0] / sample_rate
    t = torch.linspace(0, duration, message.shape[0], device=message.device)
    
    # Normalize message to [-1, 1] range for FM
    message_norm = 2 * (message - message.min()) / (message.max() - message.min() + 1e-10) - 1
    
    # Integrate message to get phase
    dt = duration / message.shape[0]
    phase = torch.cumsum(message_norm, dim=0) * dt * modulation_index
    
    # Apply FM modulation
    modulated = torch.cos(2 * torch.pi * carrier_freq * t + phase)
    
    # Normalize output
    modulated = 0.9 * modulated / torch.max(torch.abs(modulated))
    
    return modulated