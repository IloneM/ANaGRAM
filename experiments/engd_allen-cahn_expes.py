"""
ENGD Optimization.
Allen-Cahn equation equation example. Solution available at

https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/allen.cahn.html

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
from ngrad.inner import model_identity, model_del_i_factory
from classical_methods_utility import make_gram_on_model_factory

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

## THIS VALUE CAN BE OPTIMIZED
rcond = None
ep = expe_parameters = default_parameters_factory(
    input_dim=2, output_dim=1, expe_name=os.path.basename(__file__),
    n_inner_samples=30, n_boundary_samples=30, n_eval_samples=eval_points.shape[0], rcond=rcond)
expe_parameters.layer_sizes = [2, 20, 20, 20, 1]
expe_parameters.nsteps = 1001
expe_parameters.optimizer = 'engd'

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
#initial_integrator = jnp.stack((jnp.zeros(41),jnp.linspace(-1, 1, 41)), axis=1)
initial_integrator = DeterministicIntegrator(initial, ep.n_boundary_samples)
rboundary_integrator = DeterministicIntegrator(rboundary, ep.n_boundary_samples)
lboundary_integrator = DeterministicIntegrator(lboundary, ep.n_boundary_samples)

integrators = (initial_integrator, rboundary_integrator, lboundary_integrator, interior_integrator, eval_points)


test_interior_integrator = DeterministicIntegrator(interior, ep.n_inner_samples * 5)
test_initial_integrator = jnp.stack((jnp.zeros(205),jnp.linspace(-1, 1, 205)), axis=1)
# test_initial_integrator = DeterministicIntegrator(initial, ep.n_boundary_samples)
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

# gramians
model_del_0 = model_del_i_factory(argnum=0)
model_del_1 = model_del_i_factory(argnum=1)

def model_allen_cahn_eq_factory(diffusivity=1.):
    def model_flat_allen_cahn_eq(u_theta, g):
        lin_dt = model_del_0(u_theta, g)
        lin_ddx = model_del_1(u_theta, (model_del_1(u_theta, g)))

        def flat_allen_cahn_eq(x):
            u_ev = u_theta(x)
            du_ev, _ = jax.flatten_util.ravel_pytree(g(x))
            flat_dt, _ = jax.flatten_util.ravel_pytree(lin_dt(x))
            flat_ddx, unravel = jax.flatten_util.ravel_pytree(lin_ddx(x))
            return unravel(flat_dt - diffusivity * flat_ddx -5. * (du_ev - 3. * u_ev ** 2 * du_ev))

        return jax.jit(flat_allen_cahn_eq)

    def model_hessian_allen_cahn_eq(u_theta, g):
        def hessian_allen_cahn_eq(x):
            non_lin_factor = jnp.sqrt(30. * u_theta(x))
            du_ev, unravel = jax.flatten_util.ravel_pytree(g(x))
            return unravel(non_lin_factor * du_ev)

        return jax.jit(hessian_allen_cahn_eq)

    return model_flat_allen_cahn_eq, model_hessian_allen_cahn_eq

gram_on_model_factory = make_gram_on_model_factory((model_identity, model_identity,
                                                    model_identity, *model_allen_cahn_eq_factory(dconst)),
                                                   (initial_integrator, rboundary_integrator,
                                                    lboundary_integrator, interior_integrator, interior_integrator))

# differential operators
dt = lambda g: del_i(g, 0)
# dx = lambda g: del_i(g, 1)
ddx = lambda g: del_i(del_i(g, 1), 1)
def allen_cahn_operator(u):
    @jax.jit
    def op_on_u(tx):
        u_ev = u(tx)
        return dt(u)(tx) -dconst * ddx(u)(tx) - 5. * (u_ev - (u_ev ** 3))

    return op_on_u

functional_operators = dict(initial=identity_operator, rboundary=identity_operator, lboundary=identity_operator, interior=allen_cahn_operator)

seeds = jnp.array(loadtxt('./seeds-limited', dtype=int))

for seed in seeds:
    expe_parameters.seed = seed
    assistant = Assistant(
        integrators,
        functional_operators,
        expe_parameters,
        sources,
        u_star,
        test_integrators,
        make_gram_on_model=gram_on_model_factory)

    assistant.optimize()
