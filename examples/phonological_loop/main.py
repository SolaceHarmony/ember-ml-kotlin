import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy.io import wavfile
import argparse # Import argparse

from phonological_loop.models.phonological_loop_classifier import PhonologicalLoopClassifier
from phonological_loop.data.dataset import ModulationDataset
from phonological_loop.utils.audio import generate_tts_speech_placeholder, load_wav_to_torch, modulate_am_torch, modulate_fm_torch
from phonological_loop.utils.signal_processing import add_noise
from phonological_loop.utils.training import train_model, evaluate_model

def main(args): # Add args parameter
    print("Phonological Loop Classifier script initialized.")

    # --- Device Selection ---
    # Prioritize MPS (Mac), then CUDA, then fall back to CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Metal Performance Shaders) device for training")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device for training")
    else:
        device = torch.device("cpu")
        print(f"Using CPU for training (this will be slow)")
    
    print("Setting up example simulation...")
    
    # --- Parameters ---
    sample_rate = 16000
    carrier_freq = sample_rate / 8
    # noise_est_samples_main = sample_rate // 4 # Original: Used samples, likely too large
    noise_est_samples_main = 50 # Use first 50 frames for noise estimation
    num_classes = 3 # AM, FM, Noise/Other
    label_map = {0: "AM", 1: "FM", 2: "Noise"} # Map index to name
    inv_label_map = {v: k for k, v in label_map.items()} # Name to index
    target_len_samples = sample_rate * 1 # Process 1 second chunks
    
    # --- Generate Example Data (Optional) ---
    data_dir = "simulated_data"
    os.makedirs(data_dir, exist_ok=True) # Ensure directory exists

    if not args.skip_data_gen:
        print("Generating example data (using TTS - will use real voice if available)...")
        num_examples_per_class = 10 # Increased from 5 for better training
        for i in range(num_examples_per_class):
            # AM
            text_am = f"AM test {i}"
            tts_file_am, sr_am = generate_tts_speech_placeholder(text_am, filename=f"tts_am_{i}.wav")
            if tts_file_am and sr_am:
                msg_am, _ = load_wav_to_torch(tts_file_am)
                if msg_am is not None:
                    msg_am = msg_am.to(device)
                    sig_am = modulate_am_torch(msg_am, sr_am, carrier_freq)
                    noisy_sig_am = add_noise(sig_am, snr_db=random.uniform(-1, 10))
                    if noisy_sig_am.shape[0] < target_len_samples:
                         padding = target_len_samples - noisy_sig_am.shape[0]
                         noisy_sig_am = F.pad(noisy_sig_am, (noise_est_samples_main, padding - noise_est_samples_main))
                    else:
                         noise_prefix = add_noise(torch.zeros(noise_est_samples_main, device=device), snr_db=0)
                         noisy_sig_am = torch.cat((noise_prefix, noisy_sig_am), dim=0)
                    wavfile.write(os.path.join(data_dir, f"AM_{i}.wav"), sr_am, noisy_sig_am.cpu().numpy())
            # FM
            text_fm = f"FM test {i}"
            tts_file_fm, sr_fm = generate_tts_speech_placeholder(text_fm, filename=f"tts_fm_{i}.wav")
            if tts_file_fm and sr_fm:
                msg_fm, _ = load_wav_to_torch(tts_file_fm)
                if msg_fm is not None:
                    msg_fm = msg_fm.to(device)
                    sig_fm = modulate_fm_torch(msg_fm, sr_fm, carrier_freq)
                    noisy_sig_fm = add_noise(sig_fm, snr_db=random.uniform(-1, 10))
                    if noisy_sig_fm.shape[0] < target_len_samples:
                         padding = target_len_samples - noisy_sig_fm.shape[0]
                         noisy_sig_fm = F.pad(noisy_sig_fm, (noise_est_samples_main, padding - noise_est_samples_main))
                    else:
                         noise_prefix = add_noise(torch.zeros(noise_est_samples_main, device=device), snr_db=0)
                         noisy_sig_fm = torch.cat((noise_prefix, noisy_sig_fm), dim=0)
                    wavfile.write(os.path.join(data_dir, f"FM_{i}.wav"), sr_fm, noisy_sig_fm.cpu().numpy())
            # Noise Only - Create proper noise with similar power to the signals
            noise_len = target_len_samples
            # Instead of using add_noise on zeros, create noise directly
            noise_power = 1.0  # Set to a reasonable power level similar to signals
            noise_only = torch.randn(noise_len, device=device) * torch.sqrt(tensor.convert_to_tensor(noise_power, device=device))
            # Print noise shape and stats for debugging
            print(f"Noise_{i} shape: {noise_only.shape}, min: {noise_only.min()}, max: {noise_only.max()}, mean: {noise_only.mean()}, std: {noise_only.std()}")
            wavfile.write(os.path.join(data_dir, f"Noise_{i}.wav"), sample_rate, noise_only.cpu().numpy())
        print("Generated example data.")
    else:
        print("Skipping data generation as requested.")
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            print(f"Error: Data directory '{data_dir}' is empty or does not exist, but skipping generation was requested.")
            print("Please generate data first by running without --skip_data_gen or ensure the directory contains data.")
            exit()
    
    # --- Dataset and DataLoader ---
    print(f"Label map: {label_map}")
    print(f"Inverse label map: {inv_label_map}")
    
    # Check if data directory exists and contains files
    print(f"Data directory: {data_dir}")
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"Files in data directory: {files}")
    else:
        print(f"Data directory does not exist")
    
    # Create dataset with direct mapping
    dataset = ModulationDataset(data_dir, label_map={"am": 0, "fm": 1, "noise": 2}, target_len=target_len_samples)
    
    # Manually check each file in the dataset
    print("Checking each file in the dataset:")
    for i, filename in enumerate(dataset.filenames):
        class_name = filename.split('_')[0].lower()
        print(f"File: {filename}, Class name: {class_name}, Expected label: {dataset.class_to_idx.get(class_name, -1)}")
    
    # Filter out invalid samples if any were returned as None
    # Note: A proper implementation might use a collate_fn to handle this
    valid_indices = []
    for i in range(len(dataset)):
        try:
            wf, lbl = dataset[i]
            print(f"Index {i}: waveform shape={None if wf is None else wf.shape}, label={lbl}")
            if lbl != -1 and wf is not None:
                valid_indices.append(i)
        except Exception as e:
            print(f"Error processing index {i}: {e}")
    
    if len(valid_indices) < len(dataset):
        print(f"Filtered out {len(dataset) - len(valid_indices)} invalid samples.")
        # Create subset only if filtering occurred
        if len(valid_indices) > 0:
            dataset = Subset(dataset, valid_indices)
        else:
            print("Error: No valid data found after filtering. Exiting.")
            exit()
    
    if len(dataset) == 0:
         print("Error: No valid data found initially. Exiting.")
         exit()
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True) # Small batch size for demo
    
    # --- Model, Loss, Optimizer ---
    # Initialize classifier without noise_est_samples
    model = PhonologicalLoopClassifier(
        # Using default hop_length=128, window_length=512
        # Using default memory and S4 parameters
        num_classes=num_classes,
        # Use deep classifier with multiple hidden layers
        classifier_type='deep',
        classifier_hidden_dims=[256, 128, 64],  # Three hidden layers
        classifier_dropout=0.2,  # Add dropout for regularization
        classifier_hidden_dim=128 # Ensure this matches s4_d_model if S4 is used
    ).to(device) # Move model to device
    criterion = nn.CrossEntropyLoss()

    # --- Optimizer Setup ---
    # The model now uses lazy initialization, so we'll let the training function
    # handle the optimizer creation after the model is fully initialized
    optimizer = None
    
    # We'll run a dummy forward pass to initialize the model
    # This ensures all components are created before training starts
    with torch.no_grad():
        dummy_input = torch.zeros((1, target_len_samples), device=device)
        _ = model(dummy_input)
        
    # Now we can create the optimizer with the initialized parameters
    lr = 5e-4  # Default learning rate
    weight_decay = 0.01  # Default weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # --- Train ---
    # Revert epochs back to 20, as 100% was achieved quickly before
    train_model(model, dataloader, criterion, optimizer, num_epochs=20, noise_pretraining_epochs=10, device=device)
    
    # --- Evaluate ---
    # Get all test files and group by class
    test_files = {}
    for class_name in ["AM", "FM", "Noise"]:
        class_files = [f for f in os.listdir(data_dir) if f.startswith(class_name)]
        if class_files:
            test_files[class_name] = class_files
    
    if not test_files:
        print("No test files found for evaluation.")
    else:
        print("\n--- Evaluating on multiple signal types ---")
        
        # Select random examples from each class
        num_examples_per_class = 2  # Test 2 examples from each class
        all_test_files = []
        
        for class_name, files in test_files.items():
            # Randomly select files if there are enough
            selected_files = random.sample(files, min(num_examples_per_class, len(files)))
            all_test_files.extend([(os.path.join(data_dir, f), class_name) for f in selected_files])
        
        # Shuffle all selected files
        random.shuffle(all_test_files)
        
        # Track overall accuracy
        correct = 0
        total = 0
        
        # Evaluate each test file
        for test_wav_path, class_name in all_test_files:
            if os.path.exists(test_wav_path):
                test_waveform, _ = load_wav_to_torch(test_wav_path)
                if test_waveform is not None:
                    # Pad/truncate test waveform to target_len_samples
                    current_len = test_waveform.shape[0]
                    if current_len > target_len_samples: test_waveform = test_waveform[:target_len_samples]
                    elif current_len < target_len_samples: test_waveform = F.pad(test_waveform, (0, target_len_samples - current_len))
                    
                    # Get ground truth directly from the label map
                    # Extract class name from filename and normalize to match label_map
                    class_name_normalized = class_name.upper()
                    if class_name_normalized == "NOISE":
                        ground_truth = "Noise"  # Match the exact case in label_map
                    elif class_name_normalized == "AM":
                        ground_truth = "AM"
                    elif class_name_normalized == "FM":
                        ground_truth = "FM"
                    else:
                        ground_truth = class_name
                    
                    # Debug output
                    print(f"File: {os.path.basename(test_wav_path)}, Class: {class_name}, Normalized: {class_name_normalized}, Ground Truth: {ground_truth}")
                    
                    # Evaluate
                    print(f"\nTesting file: {os.path.basename(test_wav_path)}")
                    result = evaluate_model(
                        model,
                        test_waveform,
                        label_map,
                        device=device,
                        ground_truth=ground_truth,
                        learn_from_mistakes=False  # Don't learn during evaluation
                    )
                    
                    # Update accuracy stats
                    if result:
                        correct += 1
                    total += 1
        
        # Report overall accuracy
        if total > 0:
            accuracy = correct / total
            print(f"\nOverall Accuracy: {correct}/{total} = {accuracy:.2%}")
    
    print("\nExample simulation run complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--skip_data_gen',
        action='store_true',
        help='Skip the data generation step and use existing data in simulated_data/'
    )
    args = parser.parse_args()
    main(args)