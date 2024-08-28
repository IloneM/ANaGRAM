import jax.numpy as jnp
import jax
import optax
from optax._src.base import init_empty_state
from typing import Callable

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

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError('Params should not be None')
        return ls_update(params, updates), state

    return optax.GradientTransformation(init_empty_state, update_fn)

def update_from_nsteps(nsteps):
    nsteps = jnp.array(nsteps, dtype=jnp.int32)
    return jax.jit(lambda step: step >= nsteps)

def update_until_nsteps(nsteps):
    nsteps = jnp.array(nsteps, dtype=jnp.int32)
    return jax.jit(lambda step: step < nsteps)


def adam_lbfgs(switch_step, loss, steps,
              lr_adam=1e-3, args_adam=None, kwargs_adam=None, args_lbfgs=None, kwargs_lbfgs=None):
    if args_adam is None:
        args_adam = tuple()
    if kwargs_adam is None:
        kwargs_adam = dict()
    if args_lbfgs is None:
        args_lbfgs = tuple()
    if kwargs_lbfgs is None:
        kwargs_lbfgs = dict()

    kwargs_lbfgs['linesearch'] = scale_by_line_search(loss, steps)

    return optax.chain(optax.transforms.conditionally_transform(optax.adam(lr_adam, *args_adam, **kwargs_adam),
                                                                update_until_nsteps(switch_step)),
                       optax.transforms.conditionally_transform(optax.lbfgs(None, *args_lbfgs, **kwargs_lbfgs),
                                                                update_from_nsteps(switch_step)))

def lbfgs(loss, steps, args_lbfgs=None, kwargs_lbfgs=None):
    if args_lbfgs is None:
        args_lbfgs = tuple()
    if kwargs_lbfgs is None:
        kwargs_lbfgs = dict()

    kwargs_lbfgs['linesearch'] = scale_by_line_search(loss, steps)

    return optax.lbfgs(None, *args_lbfgs, **kwargs_lbfgs)
