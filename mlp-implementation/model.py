import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time


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
    W = jax.random.normal(w_key, (n_in, n_out)) / jnp.sqrt(n_in)

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

# Optimisation
def sgd_update(params, grads, learning_rate):
    """
    Updates params by taking old params and subtactiong the learning rate 
    multiplied by the gradient of loss for the params
    
    Args:
    params: Current params (list of (W, b) tuples)
    grads: Gradients (sam structuiure as params)
    learning_rate: step size

    Returns:
        Updated parameters (new list of (W, b) tuples)
    """

    return [
        (W - learning_rate * dW, b - learning_rate * db)
        for (W, b), (dW, db) in zip(params, grads)
    ]

@jit
def train_step(params, x_batch, y_batch, learning_rate):
    """
    Single step: compute loss, gradients, and update params
    Jit so we compile using xla
    
    Args:
        params: Current network parameters
        x_batch: Input batch
        y_batch: Label batch
        learning_rate: step_size

    Returns:
        Tuple of (updated_params, loss_value)
    """


    # # Compute loss and grads
    loss_value, grads = jax.value_and_grad(cross_entropy_loss)(params, x_batch, y_batch)
    
    # Update params using computed gradients
    new_params = sgd_update(params, grads, learning_rate)

    return new_params, loss_value

def create_batches(X, y, batch_size, key):
    """
    Create random sized mini batches
    
    Args:
        X: feature data (n_samples, n_features)
        y: labels (n_samples, )
        batch_size: number of samples per batch
        key: JAX PRNG key for shuffling

    Returns:
    List of (X_batch, y_batch) tuples
    """

    n_samples = X.shape[0]

    # randomly shuffle indices
    indices = jax.random.permutation(key, n_samples)

    batches = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx: end_idx]

        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        batches.append((X_batch, y_batch))

    return batches

def training_network(params, X_train, y_train, X_test, y_test,
                   epochs=10, batch_size=128, learning_rate=0.01, key=None):
    """
    Main training loop implementing mini-batch stochastic gradient descent

    - Create mini-bacthes
    - Forward pass to compute predictions
    - Loss calculation
    - Backward pass to compute gradients
    - Parameter updates

    Args:
        params: initial network params
        X_train, y_train: training data
        X_test, y_test: test data
        epochs: number of passes through the dataset
        batch_size: mini batch size
        learning_rate: step size for gradient descent
        key: JAX PRNG key

    Returns:
        Trained parameters
    """

    if key is None:
        key = jax.random.key(0)

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batches per epoch: {len(X_train) // batch_size}")
    print(f"  Using JIT compilation: True")
    print()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Split key for this epoch's shuffling
        key, subkey = jax.random.split(key)

        # Create mini batches
        batches = create_batches(X_train, y_train, batch_size, subkey)

        # Train for all batches
        epoch_losses = []
        for X_batch, y_batch in batches:
            params, loss_value = train_step(params, X_batch, y_batch, learning_rate)
            epoch_losses.append(loss_value)

        # Calculate metrics
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        train_acc = accuracy(params, X_train, y_train)
        test_acc = accuracy(params, X_test, y_test)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:2d}/{epochs} ({epoch_time:5.2f}s) | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")
    
    return params

def main():
    """Main function which demonstrates the entire ML Pipeline"""

    print("Fashion MNIST Dataset Neural Network")

    # Initialise random key
    key = jax.random.key(42)

    # Load and prepocess data
    X_train, X_test, y_train, y_test = load_Fashion_MNIST()

    # Define network architecture
    layer_sizes = [784, 128, 64, 10]
    print(f'\nNetwork Architecture')
    print(f'Layer sizes: {'->'.join(map(str, layer_sizes))}')

    # Initialise parameters
    key, init_key = jax.random.split(key)
    params = init_network_params(layer_sizes, init_key)

    # Count total parameters
    total_params = sum(W.size + b.size for (W, b) in params)
    print(f'Total parameters: {total_params}')
    print(f'Parameter breakdown')
    for i, (W, b) in enumerate(params):
        print(f'Layer {i+1}: W{W.shape} + b{b.shape} = {W.size+b.size} parameters')


    # Training the network
    training_start = time.time()
    key, training_key = jax.random.split(key)

    trained_params = training_network(
        params, X_train, y_train, X_test, y_test,
        epochs=20, batch_size=128, learning_rate=0.01,
        key=training_key
    )

    training_time = time.time() - training_start

    # Final results
    final_train_acc = accuracy(trained_params, X_train, y_train)
    final_test_acc = accuracy(trained_params, X_test, y_test)
    final_train_loss = cross_entropy_loss(trained_params, X_train, y_train)
    final_test_loss = cross_entropy_loss(trained_params, X_test, y_test)

    print(f"Training Accuracy:   {final_train_acc:.4f}")
    print(f"Test Accuracy:       {final_test_acc:.4f}")
    print(f"Training Loss:       {final_train_loss:.4f}")
    print(f"Test Loss:           {final_test_loss:.4f}")
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Time per epoch:      {training_time/10:.2f} seconds")

    # Sample predictions
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    key, sample_key = jax.random.split(key)
    sample_indices = jax.random.choice(sample_key, len(X_test), shape=(5,), replace=False)

    for idx in sample_indices:
        x_sample = X_test[idx:idx+1]
        y_true = y_test[idx]

        predictions = forward_pass(trained_params, x_sample)
        y_pred = jnp.argmax(predictions)
        confidence = predictions[0, y_pred]

        correct = "✓" if y_pred == y_true else "✗"
        print(f"{correct} True: {class_names[y_true]:12s} | "
              f"Predicted: {class_names[y_pred]:12s} | "
              f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main()