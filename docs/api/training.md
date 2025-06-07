# Training Module (ember_ml.training)

The `ember_ml.training` module provides backend-agnostic implementations of training components that work with any backend (NumPy, PyTorch, MLX). This module includes optimizers, loss functions, and metrics for training and evaluating models.

## Importing

```python
from ember_ml.training import Optimizer, SGD, Adam
from ember_ml.training import Loss, MSELoss, CrossEntropyLoss
from ember_ml.training import classification_metrics, regression_metrics
```

## Optimizers

The `ember_ml.training.optimizer` module provides optimizers for training neural networks.

### Base Optimizer

```python
class Optimizer:
    def __init__(self, learning_rate=0.01):
        """
        Initialize the optimizer.
        
        Args:
            learning_rate: Learning rate for the optimizer
        """
```

### SGD Optimizer

```python
class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        """
        Stochastic Gradient Descent optimizer.
        
        Args:
            learning_rate: Learning rate for the optimizer
            momentum: Momentum factor (default: 0.0)
            nesterov: Whether to use Nesterov momentum (default: False)
        """
```

### Adam Optimizer

```python
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        """
        Adam optimizer.
        
        Args:
            learning_rate: Learning rate for the optimizer
            beta_1: Exponential decay rate for the 1st moment estimates (default: 0.9)
            beta_2: Exponential decay rate for the 2nd moment estimates (default: 0.999)
            epsilon: Small constant for numerical stability (default: 1e-7)
        """
```

## Loss Functions

The `ember_ml.training.loss` module provides loss functions for training neural networks.

### Base Loss

```python
class Loss:
    def __call__(self, y_true, y_pred):
        """
        Compute the loss.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Loss value
        """
```

### MSE Loss

```python
class MSELoss(Loss):
    def __call__(self, y_true, y_pred):
        """
        Compute the mean squared error loss.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MSE loss value
        """
```

### Cross Entropy Loss

```python
class CrossEntropyLoss(Loss):
    def __init__(self, from_logits=False):
        """
        Cross entropy loss.
        
        Args:
            from_logits: Whether y_pred is a tensor of logits (default: False)
        """
        
    def __call__(self, y_true, y_pred):
        """
        Compute the cross entropy loss.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels or logits
            
        Returns:
            Cross entropy loss value
        """
```

## Metrics

The `ember_ml.training.metrics` module provides metrics for evaluating models.

### Classification Metrics

```python
def classification_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics:
        - accuracy: Accuracy score
        - precision: Precision score (macro-averaged)
        - recall: Recall score (macro-averaged)
        - f1: F1 score (macro-averaged)
    """
```

### Binary Classification Metrics

```python
def binary_classification_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute binary classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Threshold for binary classification (default: 0.5)
        
    Returns:
        Dictionary of metrics:
        - accuracy: Accuracy score
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - tp: True positives
        - tn: True negatives
        - fp: False positives
        - fn: False negatives
    """
```

### Confusion Matrix

```python
def confusion_matrix(y_true, y_pred, normalize=False):
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize the confusion matrix (default: False)
        
    Returns:
        Confusion matrix
    """
```

### ROC AUC

```python
def roc_auc(y_true, y_score):
    """
    Compute ROC curve and AUC.
    
    Args:
        y_true: True binary labels
        y_score: Target scores
        
    Returns:
        Tuple of (fpr, tpr, thresholds, auc)
    """
```

### Precision-Recall Curve

```python
def precision_recall_curve(y_true, y_score):
    """
    Compute precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_score: Target scores
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
```

### Average Precision Score

```python
def average_precision_score(y_true, y_score):
    """
    Compute average precision score.
    
    Args:
        y_true: True binary labels
        y_score: Target scores
        
    Returns:
        Average precision score
    """
```

### Regression Metrics

```python
def regression_metrics(y_true, y_pred):
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics:
        - mse: Mean squared error
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - r2: R-squared score
    """
```

## Example Usage

### Training a Model with Optimizers and Loss Functions

```python
from ember_ml.training import SGD, MSELoss
from ember_ml.nn.modules import Linear
from ember_ml.nn import tensor
from ember_ml import ops

# Create a simple linear model
model = Linear(10, 1)

# Create optimizer and loss function
optimizer = SGD(learning_rate=0.01)
loss_fn = MSELoss()

# Generate some data
X = tensor.random_normal((100, 10))
y = tensor.random_normal((100, 1))

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = model(X)
    
    # Compute loss
    loss = loss_fn(y, y_pred)
    
    # Compute gradients
    gradients = ops.gradients(loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
```

### Evaluating a Model with Metrics

```python
from ember_ml.training import classification_metrics, confusion_matrix
from ember_ml.nn import tensor

# Generate some predictions
y_true = tensor.convert_to_tensor([0, 1, 2, 0, 1, 2])
y_pred = tensor.convert_to_tensor([0, 2, 1, 0, 1, 2])

# Compute classification metrics
metrics = classification_metrics(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']}")
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
print(f"F1 Score: {metrics['f1']}")

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{cm}")
```

## See Also

- [Operations (ops)](ops.md): Documentation on operations for computing gradients
- [Neural Network Modules (nn.modules)](nn_modules.md): Documentation on neural network modules
- [Tensor Operations (nn.tensor)](nn_tensor.md): Documentation on tensor operations