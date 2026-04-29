import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_mnist_from_csv(train_csv_path, test_csv_path, val_split=0.1):
    """
    Load MNIST dataset from CSV files and split into train/validation/test sets.

    Args:
        train_csv_path (str): Path to the training CSV file. First column contains labels (0-9),
                             remaining 784 columns contain pixel values (0-255).
        test_csv_path (str): Path to the test CSV file with same format as train_csv_path.
        val_split (float): Fraction of training data to use for validation (default: 0.1).

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)- X_train: Training images, shape (n_train, 784), normalized to [0, 1]
            - y_train: Training labels, shape (n_train,), integers 0-9
            - X_val: Validation images, shape (n_val, 784), normalized to [0, 1]
            - y_val: Validation labels, shape (n_val,), integers 0-9
            - X_test: Test images, shape (n_test, 784), normalized to [0, 1]
            - y_test: Test labels, shape (n_test,), integers 0-9
    Note:
        - Uses random seed 42 for reproducible validation split
        - Pixel values are normalized by dividing by 255.0
        - Validation set is randomly sampled from training data
    """
    # Load training data from CSV
    train_data = pd.read_csv(train_csv_path)

    # Load test data from CSV
    test_data = pd.read_csv(test_csv_path)

    # Extract labels (first column) and features (remaining columns) from training data
    # Convert labels to int64 and normalize pixel values to [0, 1] range
    y_train_full = train_data.iloc[:, 0].to_numpy(np.int64)
    X_train_full = train_data.iloc[:, 1:].to_numpy(np.float32) / 255.0

    # Extract labels and features from test data
    y_test = test_data.iloc[:, 0].to_numpy(np.int64)
    X_test = test_data.iloc[:, 1:].to_numpy(np.float32) / 255.0

    # Calculate number of validation samples
    n_val = int(len(X_train_full) * val_split)

    # Set random seed for reproducibility
    np.random.seed(42)
    # Randomly select indices for validation set without replacement
    val_indices = np.random.choice(len(X_train_full), n_val, replace=False)

    # Create boolean mask: True for training samples, False for validation samples
    train_mask = np.ones(len(X_train_full), dtype=bool)
    train_mask[val_indices] = False

    # Split data into validation set
    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]

    # Split data into training set (remaining samples after validation split)
    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]

    # Print dataset statistics
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


class Layer:
    """
    Base class for neural network layers.

    All layer types inherit from this class and implement forward() and backward() methods.
    This provides a common interface for building neural networks.
    """

    def __init__(self):
        pass

    def forward(self, input):
        """
        Forward pass: compute layer output given input.

        Args:
            input (np.ndarray): Input data, shape (batch_size, input_dim)

        Returns:
            np.ndarray: Layer output, shape depends on layer type
        """
        return input

    def backward(self, grad_output):
        """
        Backward pass: compute gradient of loss w.r.t. layer input.

        Args:
            grad_output (np.ndarray): Gradient of loss w.r.t. layer output

        Returns:
            np.ndarray: Gradient of loss w.r.t. layer input
        """
        return grad_output


class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) activation layer.

    Applies element-wise activation function: f(x) = max(0, x)
    - Outputs x if x > 0, otherwise outputs 0
    - Introduces non-linearity to the network
    - Computationally efficient and helps mitigate vanishing gradient problem
    """

    def forward(self, input):
        """
        Forward pass: apply ReLU activation.

        Args:
            input (np.ndarray): Input data, shape (batch_size, num_features)

        Returns:
            np.ndarray: Activated output, same shape as input
                       Each element is max(0, input_element)
        """
        # Store input for use in backward pass
        self.input = input
        # Apply ReLU: output = max(0, input)
        return np.maximum(0, input)

    def backward(self, grad_output):
        """
        Backward pass: compute gradient through ReLU.

        ReLU derivative:
        - d(ReLU)/dx = 1 if x > 0
        - d(ReLU)/dx = 0 if x <= 0

        Args:
            grad_output (np.ndarray): Gradient of loss w.r.t. ReLU output

        Returns:
            np.ndarray: Gradient of loss w.r.t. ReLU input
                       grad_output is passed through where input > 0, zeroed elsewhere
        """
        # Create mask: 1 where input > 0, 0 elsewhere
        relu_grad = self.input > 0
        # Element-wise multiply: gradient flows through only where input was positive
        return grad_output * relu_grad


class Dense(Layer):
    """
    Fully connected (dense) layer with learnable weights and biases.

    Performs affine transformation: output = input @ weights + biases
    Uses He initialization for weights to prevent vanishing/exploding gradients.
    Updates parameters using gradient descent during backward pass.

    Args:
        input_units (int): Number of input features
        output_units (int): Number of output features (neurons)
        learning_rate (float): Step size for gradient descent updates (default: 0.1)
    """

    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate

        # He initialization: sample from N(0, 2/input_units)
        # This initialization helps maintain gradient variance across layers
        # Variance = 2/n_in is optimal for ReLU activations
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(
            2.0 / input_units
        )
        # Initialize biases to zero (common practice)
        self.biases = np.zeros(output_units)

    def forward(self, input):
        """
        Forward pass: compute linear transformation.

        Args:
            input (np.ndarray): Input data, shape (batch_size, input_units)

        Returns:
            np.ndarray: Output, shape (batch_size, output_units)
                       output[i, j] = sum_k(input[i, k] * weights[k, j]) + biases[j]
        """
        # Store input for use in backward pass (needed to compute weight gradients)
        self.input = input
        # Compute affine transformation: y = Wx + b
        return np.dot(input, self.weights) + self.biases

    def backward(self, grad_output):
        """
        Backward pass: compute gradients and update parameters.

        Computes three gradients using chain rule:
        1. Gradient w.r.t. weights: used to update weights
        2. Gradient w.r.t. biases: used to update biases
        3. Gradient w.r.t. input: passed to previous layer

        Then performs gradient descent update:
        - weights := weights - learning_rate * grad_weights
        - biases := biases - learning_rate * grad_biases

        Args:
            grad_output (np.ndarray): Gradient of loss w.r.t. layer output,shape (batch_size, output_units)

        Returns:
            np.ndarray: Gradient of loss w.r.t. layer input,
                       shape (batch_size, input_units)
        """
        # Gradient of loss w.r.t. weights: dL/dW = input^T @ grad_output
        # Shape: (input_units, batch_size) @ (batch_size, output_units) = (input_units, output_units)
        grad_weights = np.dot(self.input.T, grad_output)

        # Gradient of loss w.r.t. biases: dL/db = sum over batch dimension
        # Each bias affects all samples in batch, so we sum gradients across batch
        # Shape: (output_units,)
        grad_biases = np.sum(grad_output, axis=0)

        # Gradient of loss w.r.t. input: dL/dinput = grad_output @ weights^T
        # This gradient is passed to the previous layer in backpropagation
        # Shape: (batch_size, output_units) @ (output_units, input_units) = (batch_size, input_units)
        grad_input = np.dot(grad_output, self.weights.T)

        # Update parameters using gradient descent: param := param - lr * gradient
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input


def softmax_crossentropy_with_logits(logits, labels):
    """
    Compute softmax cross-entropy loss and its gradient.

    This function combines softmax activation and cross-entropy loss for numerical stability.
    Cross-entropy loss measures the difference between predicted probabilities and true labels.

    Mathematical formulation:
    - Softmax: p_i = exp(logit_i) / sum_j(exp(logit_j))
    - Cross-entropy: L = -sum_i(y_i * log(p_i)) where y_i is one-hot encoded label
    - Gradient: dL/d(logit) = (p - y) / batch_size

    Args:
        logits (np.ndarray): Raw network outputs (before softmax), shape (batch_size, num_classes)
        labels (np.ndarray): True class labels as integers, shape (batch_size,)
                            Each element is in range [0, num_classes-1]

    Returns:
        tuple: (loss, grad)
            - loss (float): Average cross-entropy loss over the batch
            - grad (np.ndarray): Gradient of loss w.r.t. logits, shape (batch_size, num_classes)
    """
    # Create one-hot encoded labels: shape (batch_size, num_classes)
    # one_hot_labels[i, j] = 1 if labels[i] == j, else 0
    batch_size = logits.shape[0]
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1

    # Compute softmax with numerical stability trick
    # Subtract max for numerical stability (prevents overflow in exp)
    # This doesn't change the result since softmax is translation-invariant
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Compute cross-entropy loss: L = -mean(sum(y * log(p)))
    # Add small epsilon (1e-9) to prevent log(0)
    # Average over batch to get mean loss
    loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size

    # Gradient of cross-entropy loss w.r.t. logits
    # For softmax + cross-entropy, gradient simplifies to: (predicted_probs - true_labels) / batch_size
    # Divide by batch_size to get average gradient (consistent with average loss)
    grad = (softmax_probs - one_hot_labels) / batch_size

    return loss, grad


def softmax(logits):
    """
    Apply softmax function to convert logits to probabilities.
    Softmax normalizes logits into a probability distribution:
    - Output values are in range (0, 1)
    - Output values sum to 1 across classes
    - Larger logits get exponentially larger probabilities

    Formula: softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

    Args:
        logits (np.ndarray): Raw scores, shape (batch_size, num_classes)

    Returns:
        np.ndarray: Probability distribution, shape (batch_size, num_classes)
                   Each row sums to 1.0
    """
    # Subtract max for numerical stability (prevents overflow in exp)
    # This doesn't change the result since softmax is translation-invariant
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    # Normalize by sum to get probabilities
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def forward(network, X):
    """
    Perform forward pass through the entire network.

    Sequentially applies each layer's forward function, passing output of one layer
    as input to the next. Stores all intermediate activations for use in backpropagation.

    Args:
        network (list): List of Layer objects representing the neural network
        X (np.ndarray): Input data, shape (batch_size, input_dim)

    Returns:
        list: Activations from each layer, where activations[i] is the output of layer iLength equals number of layers in network
              activations[-1] contains the final network output (logits)
    """
    activations = []
    input = X

    # Pass data through each layer sequentially
    for layer in network:
        input = layer.forward(input)
        # Store activation for this layer (used in backpropagation and for predictions)
        activations.append(input)

    return activations


def predict(network, X):
    """
    Get class predictions from the network.

    Performs forward pass and returns the class with highest probability for each sample.

    Args:
        network (list): List of Layer objects representing the neural network
        X (np.ndarray): Input data, shape (batch_size, input_dim)

    Returns:
        np.ndarray: Predicted class labels, shape (batch_size,)
                   Each element is an integer in range [0, num_classes-1]
    """
    # Get the output of the last layer (logits)
    logits = forward(network, X)[-1]
    # Convert logits to probabilities using softmax
    probs = softmax(logits)
    # Return the class with highest probability for each sample
    return np.argmax(probs, axis=-1)


def train(network, X, y):
    """
    Train the network on a batch of examples using backpropagation.

    Performs one complete forward-backward pass:
    1. Forward pass: compute predictions and loss
    2. Backward pass: compute gradients and update weights

    This implements the core training loop for gradient descent.

    Args:
        network (list): List of Layer objects representing the neural network
        X (np.ndarray): Input data, shape (batch_size, input_dim)
        y (np.ndarray): True labels, shape (batch_size,)

    Returns:
        float: Cross-entropy loss for this batch
    """
    # Forward pass: compute activations for all layers
    activations = forward(network, X)
    # Get final layer output (logits before softmax)
    logits = activations[-1]

    # Compute loss and initial gradient (gradient of loss w.r.t. logits)
    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)

    # Backward pass (backpropagation): propagate gradients through network
    # Start with gradient from loss function
    grad_output = grad_logits
    # Iterate through layers in reverse order (output to input)
    for i in range(len(network))[::-1]:
        layer = network[i]
        # Each layer computes gradient w.r.t. its input and updates its parameters
        grad_output = layer.backward(grad_output)

    return loss


def train_mnist_network(X_train, y_train, X_val, y_val, num_epochs=200):
    """
    Train a neural network on MNIST dataset using Gradient Descent (GD).

    Gradient Descent uses the entire dataset for each weight update,
    computing the true gradient of the loss function. This is different from
    Stochastic Gradient Descent (SGD) which uses mini-batches.

    Network architecture:
    - Input layer: 784 units (28x28 flattened images)
    - Hidden layer 1: 64 units with ReLU activation
    - Hidden layer 2: 32 units with ReLU activation
    - Output layer: 10 units (one per digit class 0-9)

    Training process:
    - Each epoch processes the entire training set once
    - Computes loss and gradients on full dataset
    - Updates all weights based on average gradient
    - Evaluates on both training and validation sets

    Args:
        X_train (np.ndarray): Training images, shape (n_train, 784)
        y_train (np.ndarray): Training labels, shape (n_train,)
        X_val (np.ndarray): Validation images, shape (n_val, 784)
        y_val (np.ndarray): Validation labels, shape (n_val,)
        num_epochs (int): Number of training epochs (default: 200)

    Returns:
        None (prints training progress)
    """
    # Initialize the network with 3 dense layers and 2 ReLU activations
    network = [
        Dense(
            784, 64
        ),  # Input layer -> Hidden layer 1: 784 input features to 64 neurons
        ReLU(),  # Activation function: introduces non-linearity
        Dense(64, 32),  # Hidden layer 1 -> Hidden layer 2: 64 to 32 neurons
        ReLU(),  # Activation function
        Dense(32, 10),  # Hidden layer 2 -> Output layer: 32 to 10 classes (digits 0-9)
    ]

    # Print network architecture for verification
    print("Network architecture:")
    for i, layer in enumerate(network):
        if isinstance(layer, Dense):
            print(
                f"Layer {i}: Dense ({layer.weights.shape[0]} -> {layer.weights.shape[1]})"
            )
        else:
            print(f"Layer {i}: {layer.__class__.__name__}")

    print(f"\nTraining on {len(X_train)} examples using Gradient Descent (full batch)")

    # Training loop - Gradient Descent uses the entire dataset per epoch
    for epoch in range(num_epochs):
        # Train on the entire dataset (full batch Gradient Descent)
        # This computes the true gradient of the loss function
        loss = train(network, X_train, y_train)

        # Evaluate training accuracy
        train_predictions = predict(network, X_train)
        train_accuracy = np.mean(train_predictions == y_train)

        # Evaluate validation accuracy (monitors generalization performance)
        val_predictions = predict(network, X_val)
        val_accuracy = np.mean(val_predictions == y_val)

        # Print progress: loss, training accuracy, and validation accuracy
        print(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )


if __name__ == "__main__":
    # Load MNIST dataset from CSV files
    # Splits training data into 90% train, 10% validation
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
        "./mnist_train.csv", "./mnist_test.csv", val_split=0.1
    )
    # Train the neural network
    train_mnist_network(X_train, y_train, X_val, y_val)
