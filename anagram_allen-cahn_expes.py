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
from ngrad.domains import Hyperrectangle, RectangleBoundary
from ngrad.integrators import DeterministicIntegrator
from anagram import identity_operator, null_source
from ngrad.utility import del_i
from anagram_assistant import *

jax.config.update("jax_enable_x64", True)
from jax.lib import xla_bridge
print('using device : {}'.format(xla_bridge.get_backend().platform))

from scipy.io import loadmat
from numpy import loadtxt

# Adapted from https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/allen.cahn.html
def gen_testdata():
    data = loadmat("./Allen_Cahn.mat")

    t = jnp.array(data["t"])
    x = jnp.array(data["x"])
    u = jnp.array(data["u"])

    xx, tt = jnp.meshgrid(x[0], t[0])
    X = jnp.vstack((jnp.ravel(tt), jnp.ravel(xx))).T
    y = u.flatten()[:, None]
    return X, y
u_star = gen_testdata()
eval_points = u_star[0]
u_solution = u_star[1][:,0]
u_star = (eval_points, u_solution)

#num_domain=8000, num_boundary=400, num_initial=800

## THIS VALUE CAN BE OPTIMIZED
rcond = None #1e-6
ep = expe_parameters = default_parameters_factory(
    input_dim=2, output_dim=1, expe_name=os.path.basename(__file__),
    n_inner_samples=30, n_boundary_samples=30, n_eval_samples=eval_points.shape[0], rcond=rcond)
expe_parameters.layer_sizes = [2, 20, 20, 20, 1]
expe_parameters.nsteps = 4001

if __name__ == '__main__':
    args_parser = create_parser()
    args = args_parser.parse_args()
    ep = expe_parameters = parse(args, expe_parameters)

# domains
interior = Hyperrectangle([[0.,1.], [-1.,1.]])
initial = RectangleBoundary([[0.,1.], [-1.,1.]], side_number=3)
rboundary = RectangleBoundary([[0.,1.], [-1.,1.]], side_number=0)
lboundary = RectangleBoundary([[0.,1.], [-1.,1.]], side_number=2)

# integrators
interior_integrator = DeterministicIntegrator(interior, ep.n_inner_samples)
# initial_integrator = jnp.stack((jnp.zeros(41),jnp.linspace(-1, 1, 41)), axis=1)
initial_integrator = DeterministicIntegrator(initial, ep.n_boundary_samples)
rboundary_integrator = DeterministicIntegrator(rboundary, ep.n_boundary_samples)
lboundary_integrator = DeterministicIntegrator(lboundary, ep.n_boundary_samples)

integrators = (initial_integrator, rboundary_integrator, lboundary_integrator, interior_integrator, eval_points)


test_interior_integrator = DeterministicIntegrator(interior, ep.n_inner_samples * 5)
test_initial_integrator = jnp.stack((jnp.zeros(205),jnp.linspace(-1, 1, 205)), axis=1) #DeterministicIntegrator(initial, ep.n_boundary_samples)
test_rboundary_integrator = DeterministicIntegrator(rboundary, ep.n_boundary_samples * 5)
test_lboundary_integrator = DeterministicIntegrator(lboundary, ep.n_boundary_samples * 5)
test_integrators = (test_initial_integrator, test_rboundary_integrator, test_lboundary_integrator, test_interior_integrator)

# initial condition
@jax.jit
def u_0(tx):
    x = tx[1]
    return (x ** 2) * jnp.cos(jnp.pi * x)
# v_u_0 = vmap(u_0, (0))

minus_one = lambda x: -1.

sources = (u_0, minus_one, minus_one, null_source)

## Diffusivity
dconst = 0.001

# differential operators
dt = lambda g: del_i(g, 0)
dx = lambda g: del_i(g, 1)
ddx = lambda g: del_i(del_i(g, 1), 1)
def allen_cahn_operator(u):
    @jax.jit
    def op_on_u(tx):
        u_ev = u(tx)
        return dt(u)(tx) -dconst * ddx(u)(tx) - 5. * (u_ev - (u_ev ** 3))

    return op_on_u

functional_operators = dict(initial=identity_operator, rboundary=identity_operator, lboundary=identity_operator, interior=allen_cahn_operator)

seeds = jnp.array(loadtxt('./seeds-limited-complement', dtype=int))

for seed in seeds:
    expe_parameters.seed = seed
    assistant = Assistant(
        integrators,
        functional_operators,
        expe_parameters,
        sources,
        u_star,
        test_integrators)

    assistant.optimize()
