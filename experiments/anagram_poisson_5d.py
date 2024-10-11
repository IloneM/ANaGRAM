"""
ANaGRAM Optimization.
Five dimensional Poisson equation example. Solution given by

u(x) = sum_{i=1}^5 sin(pi * x_i)

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
from ngrad.domains import Hyperrectangle, HypercubeBoundary
from ngrad.integrators import EvolutionaryIntegrator
from anagram import identity_operator, null_source, laplacian
from anagram_assistant import *


jax.config.update("jax_enable_x64", True)
from jax.lib import xla_bridge
print('using device : {}'.format(xla_bridge.get_backend().platform))

### THIS VALUE CAN BE OPTIMIZED
rcond = None

# problem dimension
dim = 5

expe_parameters = default_parameters_factory(input_dim=dim, output_dim=1, expe_name=os.path.basename(__file__),
                                             n_inner_samples=4000, n_boundary_samples=500, n_eval_samples=40000, rcond=rcond)
expe_parameters.nsteps = 1001
expe_parameters.layer_sizes = [dim, 64, 1]

if __name__ == '__main__':
    args_parser = create_parser()
    args = args_parser.parse_args()
    ep = expe_parameters = parse(args, expe_parameters)

# domains
interior = Hyperrectangle([(0., 1.) for _ in range(0, dim)])
boundary = HypercubeBoundary(dim)

# integrators
interior_integrator = EvolutionaryIntegrator(interior, key=jax.random.PRNGKey(0), N=ep.n_inner_samples)
boundary_integrator = EvolutionaryIntegrator(boundary, key= jax.random.PRNGKey(1), N=ep.n_boundary_samples)
eval_integrator = EvolutionaryIntegrator(interior, key=jax.random.PRNGKey(0), N=ep.n_eval_samples)

integrators = (boundary_integrator, interior_integrator, eval_integrator)

# solution
@jax.jit
def u_star(x):
    return (jnp.sum(jnp.sin(jnp.pi * x)))

# rhs
@jax.jit
def f(x):
    return -jnp.pi**2 * u_star(x)

sources = (u_star, f)

# differential operators
laplace_operator = lambda u: laplacian(u, tuple(range(dim)))

functional_operators = dict(boundary=identity_operator, interior=laplace_operator)

assistant = Assistant(
    integrators,
    functional_operators,
    expe_parameters,
    sources,
    u_star
)

assistant.optimize()
