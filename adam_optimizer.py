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
from jax.example_libraries import optimizers
from ngrad.utility import grid_line_search_factory
# from anagram import full_features_factory, full_quadratic_gradient_factory
# full_features_factory(model, functional_operators)
# true_gradient = full_quadratic_gradient_factory(model, functional_operators, sources)
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

class AdamOptimizer:
    def __init__(self: object,
                 model: Callable[[Iterable[jax.typing.ArrayLike], jax.typing.ArrayLike], jax.Array],
                 #model_seed: jax.typing.ArrayLike,
                 loss_samples: Iterable[jax.typing.ArrayLike],
                 functional_operators: Iterable[Callable[Callable[[jax.typing.ArrayLike], jax.Array], Callable[[jax.typing.ArrayLike], jax.Array]]],
                 sources: Callable[[jax.typing.ArrayLike], jax.Array]|None = None,
                 lr: float|None = 1e-3):
        self.model = model
        self.functional_operators = functional_operators
        self.lr = lr

        if sources is None:
            sources = tuple(null_source for fo in functional_operators)
        else:
            sources = tuple(null_source if s is None else s for s in sources)
        self.loss_samples = loss_samples

        # self.features = full_features_factory(model, functional_operators)
        # self.true_gradient = full_quadratic_gradient_factory(model, functional_operators, sources)

        self.losses = tuple(quadratic_loss_factory(model, fo, sa, so) for fo, sa, so in zip(functional_operators, loss_samples, sources))
        self.tot_loss = jax.jit(lambda params: sum(lo(params) for lo in self.losses))

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)
        # opt_state = opt_init(params)
        #
        # def step(step, opt_state):
        #     value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state))
        #     opt_state = opt_update(step, grads, opt_state)
        #     return value, opt_state
        #
        # for i in range(num_steps):
        #     value, opt_state = step(i, opt_state)
        
        grid = jnp.linspace(0, 30, 31)
        steps = 0.5**grid
        self.ls_update = grid_line_search_factory(self.tot_loss, steps)

        # self.rcond = rcond
        # self.nat_grad = quadratic_nat_grad_factory(model, functional_operators, sources, rcond, rcond_relative_to_bigger_sv)

    # def step(self: object,
    #          params: Iterable[jax.typing.ArrayLike],
    #          samples: Iterable[jax.typing.ArrayLike],
    #          grad_params: dict|None = None):
    #     # if grad_params is None:
    #     opt_state = self.opt_init(params)
    #     gradst = grads = jax.grad(self.tot_loss)(params)
    #     opt_state = self.opt_update(0, grads, opt_state)
    #     #nat_grad = nat_grads = self.get_params(opt_state)
    #         #self.nat_grad(params, samples, tol=self.rcond)
    #     # else:
    #         # if 'tol' not in grad_params:
    #         #     grad_params['tol'] = self.rcond
    #         # nat_grad = nat_grads = self.nat_grad(params, samples, **nat_grad_params)
    #     if isinstance(grad_params, dict) and 'return_details' in grad_params: #(nat_grads, tuple):
    #         gradst = (grads, 0., 0, 0)
    #
    #     #return self.get_params(opt_state), self.lr, gradst
    #     return *self.ls_update(params, grads), gradst

    def step(self,
             step,
             opt_state,
             params,
             grad_params: dict | None = None):
        #     return value, opt_state
        gradst = grads = jax.grad(self.tot_loss)(params)
        opt_state = self.opt_update(step, grads, opt_state)
        # nat_grad = nat_grads = self.get_params(opt_state)
        # self.nat_grad(params, samples, tol=self.rcond)
        # else:
        # if 'tol' not in grad_params:
        #     grad_params['tol'] = self.rcond
        # nat_grad = nat_grads = self.nat_grad(params, samples, **nat_grad_params)
        if isinstance(grad_params, dict) and 'return_details' in grad_params:  # (nat_grads, tuple):
            gradst = (grads, 0., 0, 0)

        return self.get_params(opt_state), self.lr, gradst, opt_state
        # return *self.ls_update(params, grads), gradst
    
    # def optimize(self: object,
    #              n_steps: int,
    #              init_params: Iterable[jax.typing.ArrayLike],
    #              samples: Iterable[jax.typing.ArrayLike]|None = None,
    #              hooks: dict[str, Callable]|None = None,
    #              grad_params: dict|None = None
    #             ):
    #     if hooks is None:
    #         hooks = dict()
    #
    #     params = init_params
    #     if samples is None:
    #         samples = self.loss_samples
    #
    #     opt_state = self.opt_init(params)
    #     # def step(step, opt_state):
    #     #     value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state))
    #     #     opt_state = opt_update(step, grads, opt_state)
    #     #     return value, opt_state
    #     #
    #     # for i in range(num_steps):
    #     #     value, opt_state = step(i, opt_state)
    #
    #     if 'before_loop' in hooks:
    #         hooks['before_loop'](self, params, samples, n_steps)
    #     for iteration in range(n_steps):
    #         if 'before_update' in hooks:
    #             grad_params = hooks['before_update'](self, params, samples, n_steps, iteration, grad_params)
    #         params, actual_step, grads = self.step(params, samples, grad_params)
    #         if 'after_update' in hooks:
    #             hooks['after_update'](self, params, samples, n_steps, iteration, actual_step, grads)
    #     if 'after_loop' in hooks:
    #         hooks['after_loop'](self, params, samples, n_steps)
    #     return params

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

        opt_state = self.opt_init(params)

        if 'before_loop' in hooks:
            hooks['before_loop'](self, params, samples, n_steps)
        for iteration in range(n_steps):
            if 'before_update' in hooks:
                grad_params = hooks['before_update'](self, params, samples, n_steps, iteration, grad_params)
            params, actual_step, grads, opt_state = self.step(iteration, opt_state, params, grad_params)
            if 'after_update' in hooks:
                hooks['after_update'](self, params, samples, n_steps, iteration, actual_step, grads)
        if 'after_loop' in hooks:
            hooks['after_loop'](self, params, samples, n_steps)
        return params
