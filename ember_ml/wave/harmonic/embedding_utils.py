from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class EmbeddingGenerator:
    """Class to handle text embedding generation using transformer models."""
    
    def __init__(self, model_name="bert-base-uncased"):
        """
        Initialize the embedding generator with a specified transformer model.
        
        Args:
            model_name (str): Name of the pretrained transformer model to use
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts using the pretrained transformer.
        
        Args:
            texts (List[str]): List of input texts to generate embeddings for
            
        Returns:
            TensorLike: Array of shape (num_texts, embedding_dim) containing the embeddings
        """
        embeddings = []
        for text in texts:
            # Tokenize and get model inputs
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use the CLS token embedding as the text representation
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embedding)
            
        return tensor.vstack(embeddings)
    
    def batch_generate_embeddings(self, texts, batch_size=32):
        """
        Generate embeddings for texts in batches to handle large datasets efficiently.
        
        Args:
            texts (List[str]): List of input texts
            batch_size (int): Number of texts to process in each batch
            
        Returns:
            TensorLike: Array of shape (len(texts), embedding_dim) containing the embeddings
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.generate_embeddings(batch_texts)
            embeddings.append(batch_embeddings)
            
        return tensor.vstack(embeddings)