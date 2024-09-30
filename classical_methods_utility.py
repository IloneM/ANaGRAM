import jax.numpy as jnp
import jax
import optax
from optax._src.base import init_empty_state
from typing import Callable

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
from ngrad.gram import nat_grad_factory, gram_factory

def grid_line_search_factory(loss, steps):
    def loss_at_step(step, params, tangent_params):
        updated_params = [(w - step * dw, b - step * db)
                          for (w, b), (dw, db) in zip(params, tangent_params)]
        return loss(updated_params)

    v_loss_at_steps = jax.vmap(loss_at_step, (0, None, None))

    @jax.jit
    def grid_line_search_update(params, tangent_params):
        losses = v_loss_at_steps(steps, params, tangent_params)
        step_size = steps[jnp.argmin(losses)]
        return jax.tree_util.tree_map(lambda g: -step_size * g, tangent_params)

    return grid_line_search_update

def scale_by_line_search(loss: Callable[[optax.Params], jax.typing.DTypeLike],
                         steps: jax.typing.ArrayLike):
    ls_update = grid_line_search_factory(loss, steps)

    @jax.jit
    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError('Params should not be None')
        return ls_update(params, updates), state

    return optax.GradientTransformation(init_empty_state, update_fn)

def simple_scale_by_zoom_linesearch(loss, *args, **kwargs):
    complicated_sbl = optax.scale_by_zoom_linesearch(*args, **kwargs)

    @jax.jit
    def update_fn(updates: optax.Updates,
                  state: optax.ScaleByZoomLinesearchState,
                  params: optax.Params,
                  *subargs, **subkwargs):
        # if len({'value', 'grad', 'value_fn'}.intersection(subkwargs.keys())) < 3:
        subkwargs['value'], subkwargs['grad'] = jax.value_and_grad(loss)(params)
        subkwargs['value_fn'] = loss
        return complicated_sbl.update(updates, state, params, *subargs, **subkwargs)

    return optax.GradientTransformation(complicated_sbl.init, update_fn)

def update_from_nsteps(nsteps):
    nsteps = jnp.array(nsteps, dtype=jnp.int32)
    return jax.jit(lambda step: step >= nsteps)

def update_until_nsteps(nsteps):
    nsteps = jnp.array(nsteps, dtype=jnp.int32)
    return jax.jit(lambda step: step < nsteps)


def adam_lbfgs(switch_step, loss, #steps,
              lr_adam=1e-3, args_adam=None, kwargs_adam=None, args_lbfgs=None, kwargs_lbfgs=None):
    if args_adam is None:
        args_adam = tuple()
    if kwargs_adam is None:
        kwargs_adam = dict()
    if args_lbfgs is None:
        args_lbfgs = tuple()
    if kwargs_lbfgs is None:
        kwargs_lbfgs = dict()

    #kwargs_lbfgs['linesearch'] = scale_by_line_search(loss, steps)
    # kwargs_lbfgs['linesearch'] = optax.scale_by_zoom_linesearch(max_linesearch_steps=15)
    kwargs_lbfgs['linesearch'] = simple_scale_by_zoom_linesearch(loss, max_linesearch_steps=15)

    return optax.chain(optax.transforms.conditionally_transform(optax.adam(lr_adam, *args_adam, **kwargs_adam),
                                                                update_until_nsteps(switch_step)),
                       optax.transforms.conditionally_transform(optax.lbfgs(None, *args_lbfgs, **kwargs_lbfgs),
                                                                update_from_nsteps(switch_step)))

def lbfgs(loss, #steps,
          args_lbfgs=None, kwargs_lbfgs=None):
    if args_lbfgs is None:
        args_lbfgs = tuple()
    if kwargs_lbfgs is None:
        kwargs_lbfgs = dict()

    #kwargs_lbfgs['linesearch'] = scale_by_line_search(loss, steps)
    # kwargs_lbfgs['linesearch'] = optax.scale_by_zoom_linesearch(max_linesearch_steps=15)
    kwargs_lbfgs['linesearch'] = simple_scale_by_zoom_linesearch(loss, max_linesearch_steps=15)

    return optax.lbfgs(None, *args_lbfgs, **kwargs_lbfgs)


def make_gram_on_model_factory(trafos, integrators):
    def make_gram_on_model(model):
        grams = [gram_factory(model, trafo, integrator) for trafo, integrator in zip(trafos, integrators)]
        return jax.jit(lambda params: sum([gram(params) for gram in grams]))
    return make_gram_on_model

def engd(loss, steps, gram):
    ls_update = grid_line_search_factory(loss, steps)
    nat_grad = nat_grad_factory(gram)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError('Params should not be None')

        return ls_update(params, nat_grad(params, updates)), state

    return optax.GradientTransformation(init_empty_state, update_fn)
