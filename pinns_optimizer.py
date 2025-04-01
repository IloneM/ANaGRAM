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

from typing import Iterable, Callable
from anagram import quadratic_gradient_factory, null_source
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)


# # Specify the class in __all__
# __all__ = ['', 'Optimizer']

def l2_square_norm(f, x, params=None):
    if params is None:
        return jnp.sum(jnp.mean(f(x) ** 2, axis=0))
    return jnp.sum(jnp.mean(f(params, x) ** 2, axis=0))
    # sum is to take into the case where f output is multidimenional


def quadratic_loss_factory(model, functional_operator, samples, source=None):
    integrated = quadratic_gradient_factory(model, functional_operator, source)
    return jax.jit(lambda params: l2_square_norm(integrated, samples, params))


class PinnsOptimizer:
    def __init__(self: object,
                 model: Callable[[Iterable[jax.typing.ArrayLike], jax.typing.ArrayLike], jax.Array],
                 loss_samples: Iterable[jax.typing.ArrayLike],
                 functional_operators: Iterable[Callable[
                     Callable[[jax.typing.ArrayLike], jax.Array], Callable[[jax.typing.ArrayLike], jax.Array]]],
                 sources: Callable[[jax.typing.ArrayLike], jax.Array] | None = None,
                 test_loss_samples: Iterable[jax.typing.ArrayLike] | None = None,
                 solver: optax.GradientTransformation|None = None):
        self.model = model
        self.functional_operators = functional_operators

        if sources is None:
            sources = tuple(null_source for fo in functional_operators)
        else:
            sources = tuple(null_source if s is None else s for s in sources)

        self.loss_samples = loss_samples
        self.losses = tuple(quadratic_loss_factory(model, fo, sa, so) for fo, sa, so in
                            zip(functional_operators, loss_samples, sources))
        self.tot_loss = jax.jit(lambda params: sum(lo(params) for lo in self.losses))

        if test_loss_samples is None:
            self.test_losses = self.losses
            self.test_tot_loss = self.tot_loss
        else:
            self.test_losses = tuple(quadratic_loss_factory(model, fo, sa, so) for fo, sa, so in
                                     zip(functional_operators, test_loss_samples, sources))
            self.test_tot_loss = jax.jit(lambda params: sum(lo(params) for lo in self.test_losses))

        self.solver = optax.adam(1e-3) if solver is None else solver

    def step(self,
             opt_state,
             params,
             grad_params: dict | None = None):
        gradst = grads = jax.grad(self.tot_loss)(params)
        updates, opt_state = self.solver.update(grads, opt_state, params)

        if isinstance(grad_params, dict) and 'return_details' in grad_params:
            gradst = (grads, 0., 0, 0)

        return optax.apply_updates(params, updates), 1., gradst, opt_state
    def optimize(self: object,
                 n_steps: int,
                 init_params: Iterable[jax.typing.ArrayLike],
                 samples: Iterable[jax.typing.ArrayLike] | None = None,
                 hooks: dict[str, Callable] | None = None,
                 grad_params: dict | None = None
                 ):
        if hooks is None:
            hooks = dict()

        params = init_params
        if samples is None:
            samples = self.loss_samples

        opt_state = self.solver.init(params)

        if 'before_loop' in hooks:
            hooks['before_loop'](self, params, samples, n_steps)
        for iteration in range(n_steps):
            if 'before_update' in hooks:
                grad_params = hooks['before_update'](self, params, samples, n_steps, iteration, grad_params)
            params, actual_step, grads, opt_state = self.step(opt_state, params, grad_params)
            if 'after_update' in hooks:
                hooks['after_update'](self, params, samples, n_steps, iteration, actual_step, grads)
        if 'after_loop' in hooks:
            hooks['after_loop'](self, params, samples, n_steps)
        return params
