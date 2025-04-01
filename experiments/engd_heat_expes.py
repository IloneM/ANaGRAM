"""
ENGD Optimization.
One dimensional heat equation example. Solution given by

u(t,x) = exp(pi**2 * t * 0.25) * sin(pi * x).

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
from anagram import identity_operator, null_source
from ngrad.utility import del_i
from anagram_assistant import *
from numpy import loadtxt
from ngrad.inner import model_identity, model_del_i_factory
from classical_methods_utility import make_gram_on_model_factory

jax.config.update("jax_enable_x64", True)
from jax.lib import xla_bridge
print('using device : {}'.format(xla_bridge.get_backend().platform))

## THIS VALUE CAN BE OPTIMIZED
rcond = None

expe_parameters = default_parameters_factory(input_dim=2, output_dim=1, expe_name=os.path.basename(__file__),
                                             n_inner_samples=30, n_boundary_samples=30, n_eval_samples=300, rcond=rcond)
expe_parameters.layer_sizes = [2, 64, 1]
expe_parameters.nsteps = 2001
expe_parameters.optimizer = 'engd'


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

test_interior_integrator = DeterministicIntegrator(interior, ep.n_inner_samples * 5)
test_initial_integrator = DeterministicIntegrator(initial, ep.n_boundary_samples * 5)
test_rboundary_integrator = DeterministicIntegrator(rboundary, ep.n_boundary_samples * 5)
test_lboundary_integrator = DeterministicIntegrator(lboundary, ep.n_boundary_samples * 5)

test_integrators = (test_initial_integrator, test_rboundary_integrator, test_lboundary_integrator, test_interior_integrator)

# initial condition
@jax.jit
def u_0(tx):
    x = tx[1]
    return jnp.sin(jnp.pi * x)

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

# gramians
# defining heat eq inner product
model_del_0 = model_del_i_factory(argnum=0)
model_del_1 = model_del_i_factory(argnum=1)

def model_heat_eq_factory(diffusivity=1.):
    def model_heat_eq(u_theta, g):
        dg_1 = model_del_0(u_theta, g)
        ddg_2 = model_del_1(u_theta, (model_del_1(u_theta, g)))

        def return_heat_eq(x):
            flat_dg_1, unravel = jax.flatten_util.ravel_pytree(dg_1(x))
            flat_ddg_2, unravel = jax.flatten_util.ravel_pytree(ddg_2(x))
            return unravel(flat_dg_1 - diffusivity * flat_ddg_2)

        return return_heat_eq

    return model_heat_eq

gram_on_model_factory = make_gram_on_model_factory((model_identity, model_identity,
                                                    model_identity, model_heat_eq_factory(expe_diffusivity)),
                                                   (lboundary_integrator, rboundary_integrator,
                                                    initial_integrator, interior_integrator))

# differential operators
dt = lambda g: del_i(g, 0)
ddx = lambda g: del_i(del_i(g, 1), 1)
def heat_operator(u):
    return lambda tx: dt(u)(tx) - expe_diffusivity * ddx(u)(tx)

functional_operators = dict(initial=identity_operator, rboundary=identity_operator, lboundary=identity_operator, interior=heat_operator)

seeds = jnp.array(loadtxt('./seeds', dtype=int))

for seed in seeds:
    expe_parameters.seed = seed
    assistant = Assistant(
        integrators,
        functional_operators,
        expe_parameters,
        sources,
        u_star,
        test_integrators,
        make_gram_on_model=gram_on_model_factory,
    )

    assistant.optimize()
