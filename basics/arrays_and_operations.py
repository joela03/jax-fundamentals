"""
JAX Fundamentals: Array Operations
Learning objectives:
- Understand JAX arrays vs NumPy arrays
- Array creation and manipulation
- Broadcasting
"""

import jax
import jax.numpy as jnp
import numpy as np

# Exercise 1.1: Array Creation
def create_arrays():
    """
    Create and compare JAX vs NumPy arrays
    
    Tasks:
    1. Create a JAX array from a list
    2. Create arrays using: zeros, ones, arange, linspace
    3. Create random arrays
    4. Check array properties (shape, dtype, device)
    """
    
    # 1 array turns an object to an array
    x = jnp.array([1.7,2.0,9])

    # 2 - zeros takes in arr size and type (can be bool)
    three_by_three_zeros = jnp.zeros(3,3)

    twod_1s_array = jnp.ones((2))
    two_by_three_1s_array = jnp.ones((2,3))

    range_array = jnp.arange(5)

    linearly_spaced_array = jnp.linspace(0, 10, 5)
    lin_space_array_excl_endpoint = jnp.linspace(0, 10, 5, endpoint=False)



# Exercise 1.2: Array Operations
def basic_operations():
    """
    Perform basic array operations
    
    Tasks:
    1. Element-wise operations (+, -, *, /)
    2. Matrix multiplication
    3. Reshaping and transposing
    4. Slicing and indexing
    5. Reductions (sum, mean, max, min)
    """

    x = jnp.array([3, 2, 1])
    y = jnp.array([4, 5, 6])
    # jnp.array takes one argument
    z = jnp.array([[1, 2, 3],
                  [4, 5, 5],
                  [7, 8, 9]])


    vector_sum = x + y
    vector_multiplication = z * y
    dot_product = jnp.dot(x, y)

    # all values in vector added
    vector_sum = jnp.sum(z)

    # all values in x axis row summed
    row_1_summed = jnp.sum(z , axis=1)

    transposed_vector = jnp.transpose(x)

    reshaped_as_9_elements = jnp.reshape(z, 9)
    reshaped_as_1_column =jnp.reshape(z, (9,1))

# Exercise 1.3: Broadcasting
def broadcasting_examples():
    """
    Understand JAX broadcasting rules
    
    Tasks:
    1. Add scalar to array
    2. Add 1D array to 2D array
    3. Element-wise operations with different shapes
    4. Implement: normalize array to mean=0, std=1
    """
    
    # Jax tries to make operation on different shappes compatible by
    # ALigning shapes from the righ, adding size 1 dimensions if needed,
    # or repeating (broadcastiong) along those dimensions

    # Scalar + array
    x = jnp.array([[1, 2, 3],
                  [4, 5, 6]])
    
    ex1 = x + 5

    # 1D + 2D
    v = jnp.array([10, 20, 30])
    ex2 = x + v

    # elment wise operations
    col = jnp.array([[1],
                     [2]])
    
    ex3 = x * col

    # Normalise
    def normalise(x):
        mean = jnp.mean(x, axis=0, keepdims=True)
        std = jnp.std(x, axis=0, keepdims=True)

        return (x - mean) / (std + 1e-8)
    
    ex4 = normalise(x)

    print(ex4)


# Exercise 1.4: Challenge - Implement Simple Functions

def softmax(x):
    """Implement softmax function (numerically stable)"""
    
    x = x - jnp.max(x)
    exp_x = jnp.exp(x)

    return exp_x/ jnp.sum(exp_x)

    

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""

    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))

if __name__ == "__main__":
        
    x = jnp.array([1, 2, 2])
    y = jnp.array([4, 5, 6])

    print(cosine_similarity(x, y))
