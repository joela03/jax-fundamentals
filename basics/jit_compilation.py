"""
JAX Fundamentals: JIT Compilation
Learning objectives:
- Understand @jit decorator
- Measure performance improvements
- Handle JIT constraints (no side effects, static args)
- Common JIT patterns
"""
import jax.numpy as jnp
from jax import jit
import time

def jit_basics():
    # Compare performance with and without JIT
    def slow_function(x):
        result = 0
        for i in range(x.shape[0]):
            result += jnp.sum(x[i] ** 2)
        return result
    
    fast_function = jit(slow_function)
    
    # Time both versions
    large_array = jnp.ones((1000, 1000))

def jit_patterns():
    # Common JIT usage patterns
    @jit
    def matrix_operations(x, y):
        return jnp.dot(x, y) + jnp.tanh(x) * jnp.sin(y)
    
    # Your tasks:
    # 1. Create functions that benefit from JIT
    # 2. Handle cases where JIT can't be used (control flow with dynamic shapes)
    # 3. Use static_argnums for functions with non-array arguments

    # 1 faster when many elmentwise operation, if theres a function with reduction + broadcasting
    # or an mlp block
    @jit
    def heavy_math(x):
        return jnp.sin(x) + jnp.tanh(x) * jnp.sqrt(x + 1)
    
    @jit
    def row_norms(x):
        return jnp.sqrt(jnp.summ(x**2, axis=1))
    
    @jit
    def mlp_layer(w, b, x):
        return jnp.tanh(jnp.dot(w, x) + b)