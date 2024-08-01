"""
ENGD Optimization.
One dimensional heat equation example. Solution given by

u(t,x) = exp(pi**2 * t * 0.25) * sin(pi * x).

"""
import sys
import os

import numpy as np

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
# from ngrad.domains import Hyperrectangle, RectangleBoundary
# from ngrad.integrators import DeterministicIntegrator
from anagram import identity_operator, null_source
from ngrad.utility import del_i
from anagram_assistant import *

jax.config.update("jax_enable_x64", True)
from jax.lib import xla_bridge
print('using device : {}'.format(xla_bridge.get_backend().platform))

burger_solution = jnp.load('./Burgers.npz')
eval_points = jnp.array(tuple((t,x) for t in burger_solution['t'][:,0] for x in burger_solution['x'][:,0]))
u_solution = burger_solution['usol'].flatten('F') #.reshape(burger_solution['t'].shape[0], burger_solution['x'].shape[0])
u_star = (eval_points, u_solution)

## THIS VALUE CAN BE OPTIMIZED
rcond = None #1e-6
expe_parameters = default_parameters_factory(input_dim=2, output_dim=1, expe_name=os.path.basename(__file__),
                                             n_inner_samples=64, n_boundary_samples=64, n_eval_samples=eval_points.shape[0], rcond=rcond)
expe_parameters.layer_sizes = [2, 32, 64, 1] #[2, 64, 64, 1]

if __name__ == '__main__':
    args_parser = create_parser()
    args = args_parser.parse_args()
    ep = expe_parameters = parse(args, expe_parameters)

# # domains
# interior = Hyperrectangle([[0.,1.], [-1.,1.]])
# initial = RectangleBoundary([[0.,1.], [-1.,1.]], side_number=3)
# rboundary = RectangleBoundary([[0.,1.], [-1.,1.]], side_number=0)
# lboundary = RectangleBoundary([[0.,1.], [-1.,1.]], side_number=2)
#
# # integrators
# interior_integrator = DeterministicIntegrator(interior, ep.n_inner_samples)
# initial_integrator = DeterministicIntegrator(initial, ep.n_boundary_samples)
# rboundary_integrator = DeterministicIntegrator(rboundary, ep.n_boundary_samples)
# lboundary_integrator = DeterministicIntegrator(lboundary, ep.n_boundary_samples)
# eval_integrator = DeterministicIntegrator(interior, ep.n_eval_samples)

#get_space_points = lambda n_samples: jnp.linspace(-1., 1., n_samples)

# def get_space_points(n_samples):
#     eff_n_samples = n_samples // 2
#     x_p = jnp.logspace(-20, 0., eff_n_samples, base=2.)
#     x_m = -x_p[::-1]
#     return jnp.concatenate((x_m, jnp.zeros(1), x_p))

def get_space_points(n_samples):
    eff_n_samples = n_samples // 4
    x_p_log = jnp.logspace(-20., -6., eff_n_samples, base=2.)
    x_m_log = -x_p_log[::-1]
    x_m = jnp.linspace(-1., -.5**6, eff_n_samples, endpoint=False)
    x_p = -x_m[::-1]
    return jnp.concatenate((x_m, x_m_log, jnp.zeros(1), x_p_log, x_p))

interior_sample = jnp.array(tuple((t,x) for t in jnp.linspace(0.,1., ep.n_inner_samples+2)[1:-1] for x in get_space_points(ep.n_inner_samples+2)[1:-1]))
x_points = get_space_points(ep.n_boundary_samples)
initial_sample = jnp.stack((jnp.zeros_like(x_points), x_points), axis=1)
rboundary_sample = jnp.stack((jnp.linspace(0.,1., ep.n_boundary_samples), jnp.full(ep.n_boundary_samples, 1.), ), axis=1)
lboundary_sample = jnp.stack((jnp.linspace(0.,1., ep.n_boundary_samples), jnp.full(ep.n_boundary_samples, -1.), ), axis=1)

integrators = (initial_sample, rboundary_sample, lboundary_sample, interior_sample, eval_points)

# initial condition
@jax.jit
def u_0(tx):
    x = tx[1]
    return -jnp.sin(jnp.pi * x)
# v_u_0 = vmap(u_0, (0))

sources = (u_0, null_source, null_source, null_source)

# solution
# @jax.jit
# def u_star(tx):
#     t = tx[0]
#     x = tx[1]
#     return jnp.exp(-jnp.pi**2 * t * expe_diffusivity) * jnp.sin(jnp.pi * x)

# Chosen Diffusivity.
## CHANGE THIS IF NEEDED
expe_viscosity = .01 / jnp.pi

# differential operators
dt = lambda g: del_i(g, 0)
dx = lambda g: del_i(g, 1)
ddx = lambda g: del_i(del_i(g, 1), 1)
def burger_operator(u):
    return jax.jit(lambda tx: dt(u)(tx) + u(tx) * dx(u)(tx) - expe_viscosity * ddx(u)(tx))

functional_operators = dict(initial=identity_operator, rboundary=identity_operator, lboundary=identity_operator, interior=burger_operator)

assistant = Assistant(
    integrators,
    functional_operators,
    expe_parameters,
    sources,
    u_star
)

assistant.optimize()
