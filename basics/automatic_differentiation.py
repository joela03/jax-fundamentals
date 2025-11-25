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

    # Definining parameters
    x = jnp.array([0.5, -1.0, 2.0])

    w1 = jnp.array([
        [0.1, -0.2, 0.3],
        [0.4,  0.1, -0.1]
    ])

    b1 = jnp.array([0.01, -0.02])

    w2 = jnp.array([0.2, -0.3])

    b2 = jnp.array(0.05)

    params = (w1, b1, w2, b2)

    # 1
    grad_wrt_x = grad(simple_network, argnums=1)(params, x)
    print("Gradient of network output w.r.t x:", grad_wrt_x)

    # 2
    grad_wrt_params= grad(simple_network, argnums=0)(params, x)
    print("Gradient of network output w.r.t params:", grad_wrt_params)
    
    # 3
    def loss(params, x, y_true):
        y_pred = simple_network(params, x)
        return (y_pred - y_true) ** 2
    
    y_true = 1.0  

    loss_val, loss_grad = value_and_grad(loss, argnums=0)(params, x, y_true)   
    print("\nLoss value:", loss_val)
    print("Gradients of loss w.r.t params:", loss_grad)

if __name__ == "__main__":
    neural_network_gradients()