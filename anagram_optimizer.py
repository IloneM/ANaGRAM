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
from anagram import quadratic_gradient_factory
import jax
import jax.numpy as jnp
from ngrad.utility import grid_line_search_factory
from anagram import quadratic_nat_grad_factory, null_source
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

class Optimizer:
    def __init__(self: object,
                 model: Callable[[Iterable[jax.typing.ArrayLike], jax.typing.ArrayLike], jax.Array],
                 #model_seed: jax.typing.ArrayLike,
                 loss_samples: Iterable[jax.typing.ArrayLike],
                 functional_operators: Iterable[Callable[Callable[[jax.typing.ArrayLike], jax.Array], Callable[[jax.typing.ArrayLike], jax.Array]]],
                 sources: Callable[[jax.typing.ArrayLike], jax.Array]|None = None,
                 rcond: int|None = None):
        self.model = model
        self.functional_operators = functional_operators

        if sources is None:
            sources = tuple(null_source for fo in functional_operators)
        else:
            sources = tuple(null_source if s is None else s for s in sources)
        self.loss_samples = loss_samples
        
        self.losses = tuple(quadratic_loss_factory(model, fo, sa, so) for fo, sa, so in zip(functional_operators, loss_samples, sources))
        self.tot_loss = jax.jit(lambda params: sum(lo(params) for lo in self.losses))
        
        grid = jnp.linspace(0, 30, 31)
        steps = 0.5**grid
        self.ls_update = grid_line_search_factory(self.tot_loss, steps)

        self.rcond = rcond
        self.nat_grad = quadratic_nat_grad_factory(model, functional_operators, sources, rcond)

    def step(self: object,
             params: Iterable[jax.typing.ArrayLike],
             samples: Iterable[jax.typing.ArrayLike],
             nat_grad_params: dict|None = None,
            ):
        if nat_grad_params is None:
            nat_grad = nat_grads = self.nat_grad(params, samples, tol=self.rcond)
        else:
            if 'tol' not in nat_grad_params:
                nat_grad_params['tol'] = self.rcond
            nat_grad = nat_grads = self.nat_grad(params, samples, **nat_grad_params)
            if isinstance(nat_grads, tuple):
                nat_grad = nat_grads[0]
                
        return *self.ls_update(params, nat_grad), nat_grads
    
    def optimize(self: object,
                 n_steps: int,
                 init_params: Iterable[jax.typing.ArrayLike],
                 samples: Iterable[jax.typing.ArrayLike]|None = None,
                 hooks: dict[str, Callable]|None = None,
                 nat_grad_params: dict|None = None
                ):
        if hooks is None:
            hooks = dict()
        
        params = init_params
        if samples is None:
            samples = self.loss_samples
        
        if 'before_loop' in hooks:
            hooks['before_loop'](self, params, samples, n_steps)
        for iteration in range(n_steps):
            if 'before_update' in hooks:
                nat_grad_params = hooks['before_update'](self, params, samples, n_steps, iteration, nat_grad_params)
            params, actual_step, nat_grads = self.step(params, samples, nat_grad_params)
            if 'after_update' in hooks:
                hooks['after_update'](self, params, samples, n_steps, iteration, actual_step, nat_grads)
        if 'after_loop' in hooks:
            hooks['after_loop'](self, params, samples, n_steps)
        return params
