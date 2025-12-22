"""
JAX Fundamentals: Vectorization with vmap
Learning objectives:
- Understand batch processing with vmap
- Apply functions to batches of data
- Combine vmap with grad and jit
- Handle different in_axes/out_axes
"""

from jax import vmap
import jax.numpy as jnp


def vmap_basics():
    # vmap takes a function that works on one example and applies it many functions at once without looping
    # It serves as Jax's replacement for manual Python batching

    # Process one example at a time
    def process_single(x):
        return x ** 2 + 1

    # Process batch efficiently
    process_batch = vmap(process_single)

    # 1. Single-example NN forward
    def single_forward(params, x):
        w1, b1, w2, b2 = params

        def calc_z(w, b, input):
            return jnp.dot(w, input) + b

        a1 = jnp.tanh(calc_z(w1, b1, x))
        out = jnp.tanh(calc_z(w2, b2, a1))
        return out

    # vmap: params unbatched, x batched along axis 0
    batched_forward = vmap(single_forward, in_axes=(None, 0))
mk
    single_data = jnp.array([1.0, 2.0, 3.0])      # shape (3,)
    batch_data  = jnp.array([
        [1.0, 2.0, 3.0],   # shape (3,)
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])                                            # shape (3, 3)

    # Define parameters with correct shapes
    w1 = jnp.array([[0.5, 0.1, -0.2],
                    [0.3, -0.4, 0.2]])   # shape (2, 3)
    b1 = jnp.array([0.1, 0.2])           # shape (2,)
    w2 = jnp.array([0.7, -0.5])          # shape (2,)
    b2 = jnp.array(0.3)                  # scalar
    params = (w1, b1, w2, b2)

    # Run batched forward
    outputs = batched_forward(params, batch_data)

    print("Batched outputs:", outputs)
    return outputs


# Standard Python main block
if __name__ == "__main__":
    vmap_basics()