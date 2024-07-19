"""
ENGD Optimization.
One dimensional heat equation example. Solution given by

u(t,x) = exp(pi**2 * t * 0.25) * sin(pi * x).

"""
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
from ngrad.domains import Square, SquareBoundary
from ngrad.integrators import DeterministicIntegrator
from anagram import identity_operator, null_source
from ngrad.utility import del_i
from anagram_assistant import *

jax.config.update("jax_enable_x64", True)
from jax.lib import xla_bridge
print('using device : {}'.format(xla_bridge.get_backend().platform))

## THIS VALUE CAN BE OPTIMIZED
rcond = None #1e-6

expe_parameters = default_parameters_factory(input_dim=2, output_dim=1, expe_name=os.path.basename(__file__),
                                             n_inner_samples=30, n_boundary_samples=30, n_eval_samples=300, rcond=rcond)
expe_parameters.layer_sizes = [2, 64, 1]

if __name__ == '__main__':
    args_parser = create_parser()
    args = args_parser.parse_args()
    ep = expe_parameters = parse(args, expe_parameters)

# domains
interior = Square(1.)
initial = SquareBoundary(1., side_number=3)
rboundary = SquareBoundary(1., side_number=0)
lboundary = SquareBoundary(1., side_number=2)

# integrators
interior_integrator = DeterministicIntegrator(interior, ep.n_inner_samples)
initial_integrator = DeterministicIntegrator(initial, ep.n_boundary_samples)
rboundary_integrator = DeterministicIntegrator(rboundary, ep.n_boundary_samples)
lboundary_integrator = DeterministicIntegrator(lboundary, ep.n_boundary_samples)
eval_integrator = DeterministicIntegrator(interior, ep.n_eval_samples)

integrators = (initial_integrator, rboundary_integrator, lboundary_integrator, interior_integrator, eval_integrator)

# initial condition
@jax.jit
def u_0(tx):
    x = tx[1]
    return jnp.sin(jnp.pi * x)
# v_u_0 = vmap(u_0, (0))

sources = (u_0, null_source, null_source, null_source)

# solution
@jax.jit
def u_star(tx):
    t = tx[0]
    x = tx[1]
    return jnp.exp(-jnp.pi**2 * t * expe_diffusivity) * jnp.sin(jnp.pi * x)

# Chosen Diffusivity.
## CHANGE THIS IF NEEDED
expe_diffusivity = .25

# differential operators
dt = lambda g: del_i(g, 0)
ddx = lambda g: del_i(del_i(g, 1), 1)
def heat_operator(u):
    return lambda tx: dt(u)(tx) - expe_diffusivity * ddx(u)(tx)

functional_operators = dict(initial=identity_operator, rboundary=identity_operator, lboundary=identity_operator, interior=heat_operator)

assistant = Assistant(
    integrators,
    functional_operators,
    expe_parameters,
    sources,
    u_star
)

assistant.optimize()
