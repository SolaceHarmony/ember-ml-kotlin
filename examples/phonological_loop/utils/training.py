import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

def train_model(model, dataloader, criterion, optimizer=None, num_epochs=10, device=None, noise_pretraining_epochs=5):
    """Train the model for a specified number of epochs."""
    # If no device is provided, select the best available device
    if device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Auto-selected MPS device for training")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Auto-selected CUDA device for training")
        else:
            device = torch.device("cpu")
            print("Auto-selected CPU for training (this will be slow)")
    
    model.to(device)
    model.train()

    # ------------------------------------------------------------------
    # Ensure the optimizer is built *after* potential layer freezing so
    # that it only sees parameters that actually require gradients.
    # If the caller passed an optimizer we scrub its param groups to
    # exclude frozen tensors; otherwise we create a fresh AdamW.
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Safeguard: if everything is frozen, automatically unfreeze the model
    # (except for parts the caller explicitly frozen later, e.g. noise filter).
    # This prevents hard crashes when the caller forgets to un‑freeze layers.
    # ------------------------------------------------------------------
    if len(trainable_params) == 0:
        print(
            "Warning: all model parameters are currently frozen. "
            "Automatically setting requires_grad=True for every parameter."
        )
        for p in model.parameters():
            p.requires_grad_(True)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        # --------------------------------------------------------------
        # If an optimizer instance was *supplied* but it currently has
        # zero live parameters, scrap it and build a fresh one based on
        # the (now‑unfrozen) parameter list. This prevents crashes when
        # main() created an optimizer before layers were unfrozen.
        # --------------------------------------------------------------
        if optimizer is not None:
            empty_groups = all(len(g["params"]) == 0 for g in optimizer.param_groups)
            if empty_groups and len(trainable_params) > 0:
                print("Rebuilding optimizer – previously passed optimizer had no "
                      "trainable tensors after auto‑unfreeze.")
                # Preserve lr / wd hyper‐params from the first (and only) group
                hyper = optimizer.param_groups[0] if optimizer.param_groups else {}
                lr = hyper.get("lr", 5e-4)
                wd = hyper.get("weight_decay", 1e-2)
                # Replace with new AdamW
                optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        if len(trainable_params) == 0:
            raise RuntimeError(
                "No trainable parameters found even after attempting to unfreeze "
                "the model. Please check your model/freezing logic."
            )

    # ------------------------------------------------------------------
    # Build optimizer if we haven't already rebuilt / been given one
    # with live parameters.
    # ------------------------------------------------------------------
    if optimizer is None:
        optimizer = optim.AdamW(trainable_params, lr=5e-4, weight_decay=1e-2)
        print(f"Optimizer constructed with {len(trainable_params)} trainable tensors.")
    else:
        # Filter out any frozen parameters that may still be present
        for idx, group in enumerate(optimizer.param_groups):
            original_len = len(group["params"])
            group["params"] = [p for p in group["params"] if p.requires_grad]
            if len(group["params"]) == 0:
                print(f"Warning: optimizer param group {idx} is empty after filtering frozen params.")
            elif len(group["params"]) != original_len:
                print(f"Optimizer param group {idx}: {original_len} -> {len(group['params'])} trainable tensors after filtering.")
    # ------------------------------------------------------------------
    
    # Add learning rate scheduler to reduce LR as training progresses
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Add learning rate warm-up scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=10
    )
    
    print(f"\n--- Starting Training on {device} ---")
    
    # Estimate noise stats from first batch if using LogNoiseFilter
    if hasattr(model, 'noise_filter_revised') and not model.noise_filter_revised.stats_estimated:
        print("Estimating noise stats from first batch...")
        # Get first batch
        for waveforms, _ in dataloader:
            waveforms = waveforms.to(device)
            # Extract features and estimate stats
            features_inst = model.feature_extractor_inst(waveforms)
            model.noise_filter_revised.estimate_noise_stats(features_inst)
            break
    
    # Phase 1: Noise Pretraining - Train only on noise samples
    if noise_pretraining_epochs > 0:
        print(f"\n--- Phase 1: Noise Pretraining ({noise_pretraining_epochs} epochs) ---")
        for epoch in range(noise_pretraining_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (waveforms, labels) in enumerate(dataloader):
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                
                # Skip invalid samples
                if torch.any(labels == -1):
                    valid_mask = labels != -1
                    if not torch.any(valid_mask):
                        continue  # Skip batch if all samples are invalid
                    waveforms = waveforms[valid_mask]
                    labels = labels[valid_mask]
                
                # Only use noise samples (label 2)
                noise_mask = labels == 2
                if not torch.any(noise_mask):
                    continue  # Skip batch if no noise samples
                
                noise_waveforms = waveforms[noise_mask]
                noise_labels = labels[noise_mask]
                
                optimizer.zero_grad()
                
                outputs = model(noise_waveforms)
                loss = criterion(outputs, noise_labels)
                
                # Add log_dt regularization if available
                if hasattr(model, 's4_layer') and hasattr(model.s4_layer, 'log_dt_reg'):
                    loss = loss + model.s4_layer.log_dt_reg
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected at Noise Pretraining Epoch {epoch+1}, Batch {batch_idx}. Skipping update.")
                    continue
                
                loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # Step the warmup scheduler for the first few iterations
                if batch_idx < 10:
                    warmup_scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Noise Pretraining Epoch [{epoch+1}/{noise_pretraining_epochs}], Avg Loss: {avg_loss:.4f}, Time: {epoch_duration:.4f}s")
    
    # Phase 2: Full Training - Train on all samples
    print(f"\n--- Phase 2: Full Training ({num_epochs} epochs) ---")
    for epoch in range(num_epochs):
        # Freeze/unfreeze noise filter based on epoch
        # No noise filter freezing/unfreezing needed anymore
        # if epoch == 0:
        #     pass
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (waveforms, labels) in enumerate(dataloader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            # Skip invalid samples
            if torch.any(labels == -1):
                valid_mask = labels != -1
                if not torch.any(valid_mask):
                    continue  # Skip batch if all samples are invalid
                waveforms = waveforms[valid_mask]
                labels = labels[valid_mask]
            
            optimizer.zero_grad()
            
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            
            # Add log_dt regularization if available
            if hasattr(model, 's4_layer') and hasattr(model.s4_layer, 'log_dt_reg'):
                loss = loss + model.s4_layer.log_dt_reg
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at Epoch {epoch+1}, Batch {batch_idx}. Skipping update.")
                continue
            
            loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # Step the warmup scheduler for the first few iterations
            if batch_idx < 10:
                warmup_scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Full Training Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, Time: {epoch_duration:.4f}s")
        
        # Step the scheduler based on the average loss
        scheduler.step(avg_loss)
    
    print("--- Training Finished ---")

def evaluate_model(model, waveform, label_map, device=None, ground_truth=None, learn_from_mistakes=False, optimizer=None, criterion=None):
    """
    Inference function for a single waveform with optional ground truth comparison and learning.
    
    Args:
        model: The model to evaluate
        waveform: Input waveform tensor
        label_map: Dictionary mapping class indices to class names
        device: Device to run inference on
        ground_truth: Optional ground truth label (index or name)
        learn_from_mistakes: Whether to update model weights if prediction is wrong
        optimizer: Optimizer for weight updates (required if learn_from_mistakes=True)
        criterion: Loss function (required if learn_from_mistakes=True)
    """
    # If no device is provided, select the best available device
    if device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Auto-selected MPS device for evaluation")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Auto-selected CUDA device for evaluation")
        else:
            device = torch.device("cpu")
            print("Auto-selected CPU for evaluation (this will be slow)")
    
    model.to(device)
    
    # Convert ground truth to index if provided as string
    ground_truth_idx = None
    ground_truth_name = None
    if ground_truth is not None:
        if isinstance(ground_truth, str):
            # Convert from name to index
            inv_label_map = {v: k for k, v in label_map.items()}
            # Handle case-sensitivity issues
            if ground_truth.upper() == "NOISE" and "Noise" in inv_label_map:
                ground_truth_idx = inv_label_map["Noise"]
                ground_truth_name = "Noise"
            else:
                # Try exact match first
                ground_truth_idx = inv_label_map.get(ground_truth, None)
                # If not found, try case-insensitive match
                if ground_truth_idx is None:
                    for k, v in inv_label_map.items():
                        if k.upper() == ground_truth.upper():
                            ground_truth_idx = v
                            ground_truth_name = k
                            break
                else:
                    ground_truth_name = ground_truth
        else:
            # Assume it's already an index
            ground_truth_idx = ground_truth
            ground_truth_name = label_map.get(ground_truth_idx, "Unknown")
    
    # Set model to eval mode unless we're learning from mistakes
    if not learn_from_mistakes:
        model.eval()
    else:
        model.train()
        if optimizer is None or criterion is None:
            print("Warning: learn_from_mistakes=True but optimizer or criterion not provided. Switching to eval mode.")
            model.eval()
            learn_from_mistakes = False
    
    waveform = waveform.to(device).unsqueeze(0)  # Add batch dimension
    
    print(f"\n--- Running Inference on {device} ---")
    print(f"Input waveform shape: {waveform.shape}")
    if ground_truth is not None:
        print(f"Ground Truth: {ground_truth_name} (Index: {ground_truth_idx})")
    
    # --- Noise Estimation in Eval ---
    # Ensure noise stats are estimated or loaded before eval
    if hasattr(model, 'noise_filter_revised') and not model.noise_filter_revised.stats_estimated:
        print("Warning: Noise stats not estimated prior to eval. Filter might be inaccurate.")
        # Optionally trigger estimation on a known noise sample or use defaults
    
    # Forward pass - use no_grad context only when not learning from mistakes
    if not learn_from_mistakes:
        with torch.no_grad():
            logits = model(waveform)  # Shape: [1, num_classes]
            probabilities = F.softmax(logits, dim=-1)
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            predicted_class_name = label_map.get(predicted_class_idx, "Unknown")
            confidence = probabilities[0, predicted_class_idx].item()
    else:
        logits = model(waveform)  # Shape: [1, num_classes]
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        predicted_class_name = label_map.get(predicted_class_idx, "Unknown")
        confidence = probabilities[0, predicted_class_idx].item()
    
    print(f"Predicted Logits: {logits.cpu()}")
    print(f"Predicted Probabilities: {probabilities.cpu()}")
    print(f"Predicted Class: {predicted_class_name} (Index: {predicted_class_idx})")
    print(f"Confidence: {confidence:.4f}")
    
    # Compare with ground truth if provided
    if ground_truth_idx is not None:
        is_correct = predicted_class_idx == ground_truth_idx
        print(f"Prediction {'CORRECT' if is_correct else 'INCORRECT'}")
        
        # Learn from mistakes if enabled and prediction is wrong
        if learn_from_mistakes and not is_correct:
            print("Learning from mistake...")
            optimizer.zero_grad()
            target = tensor.convert_to_tensor([ground_truth_idx], device=device)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            print(f"Updated model weights. Loss: {loss.item():.4f}")
    
    # Return whether the prediction was correct (for accuracy tracking)
    if ground_truth_idx is not None:
        return predicted_class_idx == ground_truth_idx
    else:
        return None