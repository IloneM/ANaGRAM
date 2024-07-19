"""
ENGD Optimization.
One dimensional heat equation example. Solution given by

u(t,x) = exp(pi**2 * t * 0.25) * sin(pi * x).

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
from ngrad.domains import Square, SquareBoundary
from ngrad.integrators import DeterministicIntegrator
#from ngrad.inner import model_identity, model_del_i_factory
#from ngrad.gram import gram_factory, nat_grad_factory
from anagram import quadratic_nat_grad_factory, identity_operator, null_source, laplacian

from ngrad.utility import grid_line_search_factory, del_i

jax.config.update("jax_enable_x64", True)
#assert np.empty(1).dtype == np.float64

from jax.lib import xla_bridge
print('using device : {}'.format(xla_bridge.get_backend().platform))

from parser import Parser

# default parameters
## Layer sizes
layer_sizes = [2, 32, 1]
## random seed
seed = 42
## model
activation = lambda x : jnp.tanh(x)
model = mlp(activation)
## rcond for the SVD
### THIS VALUE CAN BE OPTIMIZED
rcond = 1e-5
## differential operators
laplace_operator = lambda u: laplacian(u, (0,1))
functional_operators = (identity_operator, laplace_operator)

if __name__ == '__main__':
    args_parser = Parser.create_parser()
    args = args_parser.parse_args()
    # pp: parsed_params
    pp, layer_sizes, seed = Parser.parse(args, layer_sizes, seed, model, functional_operators)

# random seed
# seed = 42
key = random.PRNGKey(seed)
key, key_nn, key_np = random.split(key, 3)
#rng_np = np.random.default_rng(np.asarray(key_np))

# domains
interior = Square(1.)
boundary = SquareBoundary(1.)

# integrators
interior_integrator = DeterministicIntegrator(interior, 30)
boundary_integrator = DeterministicIntegrator(boundary, 30)
eval_integrator = DeterministicIntegrator(interior, 200)

# model
# layer_sizes = [2, 32, 1]
# activation = lambda x : jnp.tanh(x)
# model = mlp(activation)
params = init_params(layer_sizes, key_nn)
v_model = vmap(model, (None, 0))

# solution
@jit
def u_star(x):
    return jnp.prod(jnp.sin(jnp.pi * x))

# rhs
@jit
def f(x):
    return -2. * jnp.pi**2 * u_star(x)

# # differential operators
# from anagram import laplacian
# laplace_operator = lambda u: laplacian(u, (0,1))

# loss terms
# def laplacian_square(u):
#     return lambda tx: laplace_operator(u)(tx) ** 2

# compute residual
# laplace_model = lambda params: laplace(lambda x: model(params, x))
# residual = lambda params, x: (laplace_model(params)(x) + f(x))**2.
# v_residual =  jit(vmap(residual, (None, 0)))

# loss
@jit
def interior_loss(params):
    laplace_model = laplace_operator(lambda x: model(params, x))
    return interior_integrator(vmap(lambda x: (laplace_model(x) - f(x)) ** 2, (0)))

@jit
def boundary_loss(params):
    return boundary_integrator(lambda x: v_model(params, x)**2)

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
# ## THIS VALUE CAN BE OPTIMIZED
# rcond = 1e-5
nat_grad = quadratic_nat_grad_factory(model, functional_operators, (null_source, f), rcond)

last_time = time.time()
print_step = 100

# training loop
for iteration in range(501):
    nat_grads = nat_grad(params, integrator_samples, tol=rcond)
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
            f'NGD Iteration: {iteration} with loss: '
            f'{loss(params)} with error L2: {l2_error} and error H1: '
            f'{h1_error} and step: {actual_step}'
        )
