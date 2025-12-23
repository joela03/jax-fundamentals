import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_Fashion_MNIST():
    """Load and process the Fashion MNIST dataset, contains 10 classes of clothing and each
    image is 28x28 and grayscale, which gets flattened to 584 features
    
    Returns:
    Tuple of (X_train, X_test, y_train, y_test) as JAX arrays
    """

    print("Loading Fashion MNIST dataset")
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, parser='auto')

    # Convert to JAX arrays and normalise pixel values (range 0 to 1 rather than 0 to 255)
    X = jnp.array(X, dtype=jnp.float32)/255.0
    y = jnp.array(y, dtype=jnp.int32)

    # Standard 60k to 10k train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, shuffle=True
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples:{X_test.shape[0]}")
    print(f"Input dimensions: {X_train.shape[1]} (28x28 flattened)")
    print(f"Number of classes: {len(jnp.unique(y))}")

    return X_train, X_test, y_train, y_test

def init_layer_params(key, n_in, n_out):
    """Initialising parameters for a single layer using Gaussian initialision
    
        Random initialisation so that:
        - symmetry is broken and all neurons will learn differently
        - small random values prevent saturaion of activation functions initially
        - standard normal is simple and works well for small network
        
        Args:
            key: JAX PRNG key for randomness
            n_in: Number of input features
            n_out: Number of output features
            
        Returns:
            Tuple of (weights, biases)
            - weights: size n_in by n_out sapmled from normal distribution
            - biases: size n_out"""
    
    # Split key for weights (key is for randomness), w_key is a tuple and we select 
    # first value from tuple
    w_key, = jax.random.split(key, 1)

    # Sample from normal distribution, w_key is a seed for the random sequence produces
    # same key means same outputs, creayes tensor with dim n_in, n_out
    W = jax.random.normal(w_key, (n_in, n_out))

    # Biases are initalised to zero
    b = jnp.zeros(n_out)

    return W, b

def init_network_params(layer_sizes, key):
    """Initialises all network parameters
    
        Args: 
            layer_sizes: list of integers (Input, Hidden, Hidden, Output)
            key: JAX PRNG key
        
        Returns:
            List of weights and biases for each layer"""
    
    # Generate a key for each set of parameters needed
    keys = jax.random.split(key, len(layer_sizes) - 1)

    params = []

    # Pair dims of consecutive layers for parameter generation
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        W, b = init_layer_params(keys[i], n_in, n_out)
        params.append((W, b))

    return params

# Activation FUnctions

def relu(x):
    """Activation function that introduces non linearity so that the model can learn"""

    return jnp.maximum(0, x)

def softmax(x):
    """"Converts logits to probability distributionb (outputs sum to 1) and range 0, 1
    
        Args:
            x: Logits (batch_size, num_classes)
        Returns:
            Probabilities (batch_size, num_classes)"""
    
    exp_x = jnp.exp(x-jnp.max(x, axis=-1, keepdims=True))
    return exp_x/ jnp.sum(exp_x, axis=-1, keepdims=True)

def forward_pass(params, x):
    """Forward propagation through the network
    
    Args:
        params: List of Weight and Bias tuples
        x: Input data of shape (batch_size, input_dim)
        
    Returns:
        Output probabilities of shape (batch_sixe, num_classes)
    """

    activations = x

    # Pass through all layers bar the last one
    for W, b in params[:-1]:
        # multiply input by Weight and bias
        z = activations @ W + b

        # non-linear activation
        activations = relu(z)

    # final output layer with softmax
    W_final, b_final = params[-1]
    logits = activations @ W_final + b_final

    return softmax(logits)


# Loss function
def cross_entropy_loss(params, x_batch, y_batch):
    """Loss function for multi class classification
    
        Formula: L = -SUM(y_true*log(y_pred)
        
        Args:
            params: network parameters
            x_batch: input batch of shape (batch_size, input_dim)
            y_batch: True labels of shape (batch size, )

        Returns:
            Scalar loss value (mean of batch)
        """
    
    # Forward pass to get predictions
    predictions = forward_pass(params, x_batch)

    # Covert int labels to one hot encoding
    y_onehot = jax.nn.one_hot(y_batch, num_classes=10)

    # Cross-entropy
    loss = -jnp.sum(y_onehot * jnp.log(predictions + 1e-8), axis=-1)

    return jnp.mean(loss)


# Metrics
def accuracy(params, x, y):
    """"Calculating classification accuracy by dividing correct predictions
        by tot predictions
        
        Args:
            params: network parameters
            x: Input data
            y: True labels

        Returns:
            Scalar accuracy [0, 1]
        """
    
    predictions = forward_pass(params, x)
    predicted_classes = jnp.argmax(predictions, axis=-1)

    return jnp.mean(predicted_classes == y)