import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
# Removed: import torch
from sklearn.metrics.pairwise import cosine_similarity
from ember_ml import ops
# Ensure stats ops are accessible if ops.stats.mean is used later
# from ember_ml.ops import stats # Or access via ops.stats.mean
from ember_ml.nn import tensor
def harmonic_wave(params, t, batch_size):
    """
    Generate a harmonic wave based on parameters.
    Handles batch processing for multiple embeddings.
    """
    harmonics = []
    for i in range(batch_size):
        amplitudes, frequencies, phases = tensor.split_tensor(params[i], 3)
        harmonic = ops.multiply(
            amplitudes[:, None], 
            ops.sin(
            ops.add(
                ops.multiply(
                ops.multiply(2, ops.pi),
                ops.multiply(frequencies[:, None], t)
                ),
                phases[:, None]
            )
            )
        )
        harmonics.append(harmonic.sum(axis=0))
    return tensor.vstack(harmonics)
from transformers import AutoTokenizer, AutoModel

# Load transformer model and tokenizer
model_name = "bert-base-uncased"  # Replace with desired transformer model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Generate embeddings
def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts using a pretrained transformer.

    NOTE: This function currently requires PyTorch and the 'transformers'
          library. It is NOT fully backend-agnostic due to the direct
          dependency on the transformer model's PyTorch implementation.

    Returns:
        EmberTensor: Tensor of shape (num_texts, embedding_dim) using the
                     current Ember ML backend.
    """
    # This part remains PyTorch-specific due to the transformers library
    try:
        import torch # Keep torch import local to this function
        import numpy as np # Needed temporarily for vstack
        embeddings_list_np = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad(): # Ensure no gradients are computed
                 outputs = model(**inputs)
            # Use the CLS token embedding, convert to numpy
            cls_embedding_np = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings_list_np.append(cls_embedding_np)
        # Convert the final list of numpy arrays to an EmberTensor
        # vstack might handle this, but explicit conversion is safer
        embeddings_np = tensor.vstack(embeddings_list_np) # Need numpy temporarily
        return tensor.convert_to_tensor(embeddings_np)
    except ImportError:
        raise ImportError("PyTorch and transformers library are required for generate_embeddings function.")
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        # Return an empty tensor or handle error appropriately
        # Returning None might be better to signal failure
        return None # Or raise the exception

# Example texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "AI is transforming the world of technology.",
    "Deep learning enables powerful language models."
]

# Generate embeddings
embeddings = generate_embeddings(texts)
def map_embeddings_to_harmonics(embeddings):
    """
    Initialize harmonic parameters for all embeddings in a batch.
    """
    batch_size, embedding_dim = embeddings.shape
    params = []
    for i in range(batch_size):
        params.append(tensor.random_normal(ops.multiply(3, embedding_dim)))  # Amplitudes, Frequencies, Phases
    return tensor.vstack(params)

def loss_function(params, t, target_embedding):
    """
    Compute the loss between the target embedding and the generated harmonic wave.
    Uses Mean Squared Error (MSE) as the metric.
    """
    # Generate harmonic wave for the given parameters
    amplitudes, frequencies, phases = tensor.split_tensor(params, 3)
    harmonic = (
        amplitudes[:, None] * ops.sin(2 * ops.pi * frequencies[:, None] * t + phases[:, None])
    ).sum(axis=0)
    
    # Compute MSE loss using ops
    diff = ops.subtract(target_embedding, harmonic)
    squared_diff = ops.square(diff) # Or ops.power(diff, 2)
    loss = ops.stats.mean(squared_diff) # Use ops.stats.mean
    return loss


def compute_gradients(params, t, target_embedding, epsilon=1e-5):
    """
    Compute numerical gradients for the harmonic parameters using finite differences.
    """
    gradients = tensor.zeros_like(params)
    for i in range(len(params)):
        params_step_plus = params.copy() # Use separate copies for clarity
        params_step_minus = params.copy()

        # Positive perturbation using ops.add
        # Note: Direct indexing and modification might need backend-specific handling
        # or tensor.scatter/slice_update if params is an EmberTensor.
        # Assuming direct modification works for now, but this is a potential issue.
        params_step_plus[i] = ops.add(params_step_plus[i], epsilon)
        loss_plus = loss_function(params_step_plus, t, target_embedding)

        # Negative perturbation using ops.subtract and ops.multiply
        two_epsilon = ops.multiply(2.0, epsilon) # Calculate 2*epsilon once
        params_step_minus[i] = ops.subtract(params_step_minus[i], two_epsilon)
        loss_minus = loss_function(params_step_minus, t, target_embedding)

        # Compute gradient using ops
        loss_diff = ops.subtract(loss_plus, loss_minus)
        denominator = ops.multiply(2.0, epsilon) # Reuse calculation
        gradients[i] = ops.divide(loss_diff, denominator)
    return gradients


def train_harmonic_embeddings(embeddings, t, batch_size, learning_rate=0.01, epochs=100):
    """
    Train harmonic wave parameters to match transformer embeddings.
    Handles multiple embeddings in batch.
    """
    params = map_embeddings_to_harmonics(embeddings)  # Random initialization
    for epoch in range(epochs):
        total_loss = 0
        for i in range(batch_size):
            # Compute loss
            loss = loss_function(params[i], t, embeddings[i])
            
            # Compute gradients
            gradients = compute_gradients(params[i], t, embeddings[i])
            
            # Update parameters using ops
            update_step = ops.multiply(learning_rate, gradients)
            # Assuming direct modification works, see note in compute_gradients
            params[i] = ops.subtract(params[i], update_step)

            # Accumulate loss using ops.add
            # Ensure total_loss is initialized as a tensor or 0.0
            if i == 0 and epoch == 0: # Initialize total_loss correctly on first step
                 total_loss = loss # Assign first loss
            else:
                 total_loss = ops.add(total_loss, loss)

        # Calculate average loss using ops.divide
        avg_loss = ops.divide(total_loss, tensor.convert_to_tensor(float(batch_size)))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {tensor.item(avg_loss)}") # Use tensor.item() to print scalar loss
    return params


# Visualize embeddings vs harmonic waves
def visualize_embeddings(target, learned):
    """
    Visualize target embeddings and learned harmonic embeddings.
    """
    plt.figure(figsize=(12, 6))

    # Plot target embeddings
    plt.subplot(211)
    plt.imshow(target, aspect="auto", cmap="viridis")
    plt.title("Target Embeddings")
    plt.colorbar()

    # Plot learned harmonic embeddings (reshaped)
    plt.subplot(212)
    plt.imshow(learned, aspect="auto", cmap="viridis")
    plt.title("Learned Harmonic Embeddings")
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    
    if __name__ == "__main__":
        # Remove numpy import
        # import numpy as np

        # Generate time steps using tensor.linspace
        # Assuming embeddings is already an EmberTensor or compatible
        num_time_steps = tensor.shape(embeddings)[1]
        t = tensor.linspace(0.0, 5.0, num_time_steps) # Use tensor.linspace

        # Train harmonic embeddings
        batch_size = embeddings.shape[0]  # Number of embeddings (batch size)
        params = train_harmonic_embeddings(embeddings, t, batch_size)
    
        # Generate learned harmonic waves
        learned_harmonic_wave = harmonic_wave(params, t, batch_size)
    
        # Reshape learned harmonic wave to match embeddings
        if learned_harmonic_wave.shape == embeddings.shape:
            # Use tensor.reshape function
            learned_harmonic_wave = tensor.reshape(learned_harmonic_wave, embeddings.shape)
        else:
            raise ValueError(
                f"Shape mismatch: learned wave shape {learned_harmonic_wave.shape}, "
                f"expected {embeddings.shape}"
            )
    
        # Visualize the results
        visualize_embeddings(embeddings, learned_harmonic_wave)