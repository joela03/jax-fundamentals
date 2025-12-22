import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_Fashion_MNIST():
    """Load and process the Fashion MNIST dataset, contains 10 classes of clothing and each
    image is 28x28 and grayscale, which gets flattened to 584 features"""

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