"""ENGD Optimization.
One dimensional heat equation example. Solution given by

u(t,x) = exp(pi**2 * t * 0.25) * sin(pi * x).

"""

import os
import sys

if "__file__" in globals():
    # If the code is running in a script, use the directory of the script file
    subfolder_path = os.path.join(
        os.path.dirname(__file__), "Natural-Gradient-PINNs-ICML23"
    )
else:
    # If the code is running interactively, use the current working directory
    subfolder_path = os.path.join(os.getcwd(), "Natural-Gradient-PINNs-ICML23")
# Add the subfolder to the system path
sys.path.append(subfolder_path)

import jax
import jax.numpy as jnp

# from ngrad.gram import gram_factory, nat_grad_factory
from anagram import identity_operator, null_source, quadratic_nat_grad_factory
from jax import grad, jit, random, vmap
from ngrad.domains import Square, SquareBoundary
from ngrad.integrators import DeterministicIntegrator
from ngrad.models import init_params, mlp
from ngrad.utility import del_i, grid_line_search_factory

jax.config.update("jax_enable_x64", True)
# assert np.empty(1).dtype == np.float64

from jax.lib import xla_bridge

print(f"using device : {xla_bridge.get_backend().platform}")

# random seed
seed = 42
key = random.PRNGKey(seed)
key, key_nn, key_np = random.split(key, 3)
# rng_np = np.random.default_rng(np.asarray(key_np))

# domains
interior = Square(1.0)
initial = SquareBoundary(1.0, side_number=3)
rboundary = SquareBoundary(1.0, side_number=0)
lboundary = SquareBoundary(1.0, side_number=2)

# integrators
interior_integrator = DeterministicIntegrator(interior, 30)
initial_integrator = DeterministicIntegrator(initial, 30)
rboundary_integrator = DeterministicIntegrator(rboundary, 30)
lboundary_integrator = DeterministicIntegrator(lboundary, 30)
eval_integrator = DeterministicIntegrator(interior, 300)

# model
activation = lambda x: jnp.tanh(x)
layer_sizes = [2, 64, 1]
params = init_params(layer_sizes, key_nn)
model = mlp(activation)
v_model = vmap(model, (None, 0))


# initial condition
def u_0(tx):
    x = tx[1]
    return jnp.sin(jnp.pi * x)


v_u_0 = vmap(u_0, (0))

# Chosen Diffusivity.
## CHANGE THIS IF NEEDED
expe_diffusivity = 0.25


# solution
def u_star(tx):
    t = tx[0]
    x = tx[1]
    return jnp.exp(-(jnp.pi**2) * t * expe_diffusivity) * jnp.sin(jnp.pi * x)


# differential operators
dt = lambda g: del_i(g, 0)
ddx = lambda g: del_i(del_i(g, 1), 1)


def heat_operator(u):
    return lambda tx: dt(u)(tx) - expe_diffusivity * ddx(u)(tx)


# loss terms
def heat_operator_square(u):
    return lambda tx: heat_operator(u)(tx) ** 2


@jit
def loss_interior(params):
    heat_model = heat_operator_square(lambda tx: model(params, tx))
    return interior_integrator(vmap(heat_model, (0)))


@jit
def loss_l_boundary(params):
    return lboundary_integrator(lambda tx: v_model(params, tx) ** 2)


@jit
def loss_r_boundary(params):
    return rboundary_integrator(lambda tx: v_model(params, tx) ** 2)


@jit
def loss_boundary(params):
    return loss_l_boundary(params) + loss_r_boundary(params)


@jit
def loss_initial(params):
    return initial_integrator(lambda tx: (v_u_0(tx) - v_model(params, tx)) ** 2)


@jit
def loss(params):
    return loss_interior(params) + loss_boundary(params) + loss_initial(params)


# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)

# errors
error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))
v_error_abs_grad = vmap(
    lambda x: jnp.dot(grad(error)(x), grad(error)(x)) ** 0.5,
)


def l2_norm(f, integrator):
    return integrator(lambda x: (f(x)) ** 2) ** 0.5


# extract samples from intergrators
def make_integrator_sample(integrator):
    return integrator._x


integrator_samples = tuple(
    make_integrator_sample(integrator)
    for integrator in (
        lboundary_integrator,
        rboundary_integrator,
        initial_integrator,
        interior_integrator,
    )
)

# constructor of the anagram natural gradient
## THIS VALUE CAN BE OPTIMIZED
rcond = 1e-6
nat_grad = quadratic_nat_grad_factory(
    model,
    (identity_operator, identity_operator, identity_operator, heat_operator),
    rcond,
    (null_source, null_source, u_0, null_source),
)

# training loop
for iteration in range(2000):
    nat_grads = nat_grad(params, integrator_samples)
    params, actual_step = ls_update(params, nat_grads)

    if iteration % 100 == 0:
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)

        print(
            f"Seed: {seed} NGD Iteration: {iteration} with loss: "
            f"{loss(params)} with error L2: {l2_error} and error H1: "
            f"{h1_error} and step: {actual_step}",
        )
