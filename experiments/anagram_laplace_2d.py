"""
ANaGRAM Optimization.
Two dimensional Laplace equation example. Solution given by

u(x,y) = sin(pi*x) * sin(py*y).

"""
import sys
import os

if '__file__' in globals():
    # If the code is running in a script, use the directory of the script file
    subfolder_paths = (os.path.join(os.path.dirname(__file__), '..', 'Natural-Gradient-PINNs-ICML23'),
                       os.path.join(os.path.dirname(__file__), '..'))
else:
    # If the code is running interactively, use the current working directory
    subfolder_paths = (os.path.join(os.getcwd(), '..', 'Natural-Gradient-PINNs-ICML23'),
                       os.path.join(os.getcwd(), '..'))
# Add the subfolder to the system path
for subfolder_path in subfolder_paths:
    sys.path.append(subfolder_path)

import jax
import jax.numpy as jnp
from ngrad.domains import Square, SquareBoundary
from ngrad.integrators import DeterministicIntegrator
from anagram import identity_operator, null_source, laplacian
from anagram_assistant import *


jax.config.update("jax_enable_x64", True)
from jax.lib import xla_bridge
print('using device : {}'.format(xla_bridge.get_backend().platform))

### THIS VALUE CAN BE OPTIMIZED
rcond = 1e-6

expe_parameters = default_parameters_factory(input_dim=2, output_dim=1, expe_name=os.path.basename(__file__),
                                             n_inner_samples=30, n_boundary_samples=30, n_eval_samples=200, rcond=rcond)
expe_parameters.nsteps = 2001
expe_parameters.rcond_relative_to_bigger_sv = False

if __name__ == '__main__':
    args_parser = create_parser()
    args = args_parser.parse_args()
    expe_parameters = parse(args, expe_parameters)

# domains
interior = Square(1.)
boundary = SquareBoundary(1.)

# integrators
interior_integrator = DeterministicIntegrator(interior, expe_parameters.n_inner_samples)
boundary_integrator = DeterministicIntegrator(boundary, expe_parameters.n_boundary_samples)
eval_integrator = DeterministicIntegrator(interior, expe_parameters.n_eval_samples)

integrators = (boundary_integrator, interior_integrator, eval_integrator)

# differential operators
laplace_operator = lambda u: laplacian(u, (0,1))
functional_operators = dict(boundary=identity_operator, interior=laplace_operator)

# rhs
@jax.jit
def f(x):
    return -2. * jnp.pi**2 * u_star(x)

sources = (null_source, f)

# solution
@jax.jit
def u_star(x):
    return jnp.prod(jnp.sin(jnp.pi * x))

assistant = Assistant(
    integrators,
    functional_operators,
    expe_parameters,
    sources,
    u_star
)

assistant.optimize()
