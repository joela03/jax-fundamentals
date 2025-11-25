"""
JAX Fundamentals: Automatic Differentiation
Learning objectives:
- Understand grad() function
- Compute gradients for different functions
- Handle multiple arguments with argnums
- Use value_and_grad for efficiency
"""

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad

def gradient_basics():
    # 1. Basic gradient
    def f(x):
        return x**2 + 3*x + 1
    
    df_dx = grad(f)
    print(f"f'(2) = {df_dx(2.0)}") 
    
    # 2. Multiple arguments
    def loss(w, b, x, y):
        return (w * x + b - y)**2
    
    # argnums specifies whuch input we differentiate w.r.t
    dloss_dw = grad(loss, argnums=0)  # derivative w.r.t w
    dloss_db = grad(loss, argnums=1)  # derivative w.r.t b
    
    # 3. value_and_grad - get both value and gradient of fucntion at point
    value, grads = value_and_grad(loss)(1.0, 0.5, 2.0, 3.0)

def neural_network_gradients():
    # Implement gradient calculations for a simple neural network
    def simple_network(params, x):
        w1, b1, w2, b2 = params
        h = jnp.tanh(jnp.dot(w1, x) + b1)
        return jnp.dot(w2, h) + b2
    
    # Your tasks:
    # 1. Compute gradient of network output w.r.t input x
    # 2. Compute gradient of network w.r.t parameters
    # 3. Implement a loss function and compute its gradients