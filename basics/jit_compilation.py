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
    
    #2 jit doesnt like changing shapes or data dependent loops
    @jit
    def bad(x, n):
        out = 0
        # n is dynamic at runtime
        for i in range(n): 
            out += x[i]
        return out
    
    # fix by using static argnums
    bad_fixed = jit(bad, static_argnums=1)
    # now n is treated as a compile-yime constant

    @jit 
    def dynamic_shape(x):
        # shape can change as x is a variable
        y = jnp.ones((x,x))
        return y

    dynamic_shape_fixed = jit(dynamic_shape, static_argnums=0)