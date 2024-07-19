"""
ANaGRAM Optimization.
Five dimensional Poisson equation example. Solution given by

u(x) = sum_{i=1}^5 sin(pi * x_i)

"""
import time
start_time = time.time()

import sys
import os

if '__file__' in globals():
    # If the code is running in a script, use the directory of the script file
    subfolder_path = os.path.join(os.path.dirname(__file__), 'Natural-Gradient-PINNs-ICML23')
else:
    # If the code is running interactively, use the current working directory
    subfolder_path = os.path.join(os.getcwd(), 'Natural-Gradient-PINNs-ICML23')
# Add the subfolder to the system path
sys.path.append(subfolder_path)

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from jax.flatten_util import ravel_pytree

from ngrad.models import mlp, init_params
from ngrad.domains import Hyperrectangle, HypercubeBoundary
from ngrad.integrators import EvolutionaryIntegrator
from anagram import quadratic_nat_grad_factory, identity_operator, null_source, laplacian

from ngrad.utility import grid_line_search_factory, del_i

jax.config.update("jax_enable_x64", True)

from jax.lib import xla_bridge
print('using device : {}'.format(xla_bridge.get_backend().platform))

# random seed
seed = 0

# domains
dim = 5
interior = Hyperrectangle([(0., 1.) for _ in range(0, dim)])
boundary = HypercubeBoundary(dim)

# integrators
interior_integrator = EvolutionaryIntegrator(interior, key=random.PRNGKey(0), N=4000)
boundary_integrator = EvolutionaryIntegrator(boundary, key= random.PRNGKey(1), N=500)
eval_integrator = EvolutionaryIntegrator(interior, key=random.PRNGKey(0), N=40000)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [dim, 64, 1]
params = init_params(layer_sizes, random.PRNGKey(seed))
model = mlp(activation)
v_model = vmap(model, (None, 0))

# solution
@jit
def u_star(x):
    return (jnp.sum(jnp.sin(jnp.pi * x)))
v_u_star = vmap(u_star, (0))


# rhs
@jit
def f(x):
    return -jnp.pi**2 * u_star(x)

# differential operators
laplace_operator = lambda u: laplacian(u, tuple(range(dim)))

# loss
@jit
def interior_loss(params):
    laplace_model = laplace_operator(lambda x: model(params, x))
    return 0.5 * interior_integrator(vmap(lambda x: (laplace_model(x) - f(x)) ** 2, (0)))

@jit
def boundary_loss(params):
    return (
        0.5 * boundary_integrator(lambda x: (v_model(params, x) - v_u_star(x))**2.)
    )

@jit
def loss(params):
    return interior_loss(params) + boundary_loss(params)

# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)

# errors
error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))
v_error_abs_grad = vmap(
        lambda x: jnp.dot(grad(error)(x), grad(error)(x))**0.5
        )

def l2_norm(f, integrator):
    return integrator(lambda x: (f(x))**2)**0.5   

# extract samples from intergrators
def make_integrator_sample(integrator):
    return integrator._x

integrator_samples = tuple(make_integrator_sample(integrator) for integrator in (boundary_integrator, interior_integrator))

# constructor of the anagram natural gradient
## THIS VALUE CAN BE OPTIMIZED
rcond = None
nat_grad = quadratic_nat_grad_factory(model, (identity_operator, laplace_operator), (u_star, f), rcond)

last_time = time.time()
print_step = 50

# training loop
for iteration in range(501):
    nat_grads = nat_grad(params, integrator_samples)
    params, actual_step = ls_update(params, nat_grads)

    if iteration % print_step == 0:
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)
        
        # Time calculation
        actual_time = time.time()
        loop_time = (actual_time - last_time) / print_step
        elapsed_time = actual_time - start_time
        last_time = actual_time

        print(
	    f'Seed: {seed} Total time elapsed: {elapsed_time:.2f}s Mean time per step : {loop_time:.2f}s '
            f'NG Iteration: {iteration} with loss: '
            f'{loss(params)} L2 error: {l2_error} H1 error: '
            f'{h1_error} and step: {actual_step}'
        )
