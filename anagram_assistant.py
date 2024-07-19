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

import argparse
from tensorflow import summary as tfsum
from datetime import datetime
import re
from typing import Iterable, Callable
from anagram import features_factory, full_features_factory, identity_operator, pre_quadratic_gradient_factory, quadratic_gradient_factory
from matplotlib import colors
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from anagram_optimizer import l2_square_norm, Optimizer
from anagram_logging_tools import write_to_tensorboard, write_weights, plot_NTK, plot_NNTK #, plot_Green
from ngrad.models import mlp, init_params
import time

# Specify the class in __all__
__all__ = ['default_parameters_factory', 'create_parser', 'parse', 'Assistant']


# source : https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class Parameters(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Default Parameters
def default_parameters_factory(input_dim, output_dim,
                               expe_name,
                               n_inner_samples, n_boundary_samples, n_eval_samples,
                               rcond=None):
    return Parameters(
        dict(layer_sizes=[input_dim, 32, output_dim], expe_name=expe_name, expe_path=None, tensorboard_path=None,
             save_final_weights=False, weights_verbosity=0, tb_verbosity=0, NNTK_verbosity=0, NTK_verbosity=0, Green_verbosity=0,
             verbosity=100, nsteps=501, seed=42, rcond=rcond, rcond_relative_to_bigger_sv=True, n_inner_samples=n_inner_samples,
             n_boundary_samples=n_boundary_samples, n_eval_samples=n_eval_samples)
    )

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-ls", "--layer_sizes",
                        help="Defines the MLP architecture as a sequence of layers sizes",
                        nargs='+', type=int)

    parser.add_argument("-exp", "--expe_name",
                        help="Name of the experiment",
                        type=str)

    parser.add_argument("-p", "--path",
                        help="Path into which the experiments potential outputs should be stored",
                        type=str)

    parser.add_argument("-tbp", "--tensorboard_path",
                        help="Path into which the tensorboard logs should be stored",
                        type=str)

    parser.add_argument("-sfw", "--save_final_weights",
                        help="Save final weights of the neural network",
                        action="store_true")

    parser.add_argument("-lw", "--log_weights",
                        help="Save weights of the neural network every n steps [0 means never]",
                        type=int)

    parser.add_argument("-tb", "--tensorboard",
                        help="Store the training metrics in tensorboard every n steps [0 means never]",
                        type=int)

    parser.add_argument("-NNTK", "--NNTK_plot",
                        help="Plot the Natural Neural Tangent Kernel of the network every n steps [0 means never]",
                        type=int)

    parser.add_argument("-NTK", "--NTK_plot",
                        help="Plot the Neural Tangent Kernel of the network every n steps [0 means never]",
                        type=int)

    parser.add_argument("-Green", "--Green_plot",
                        help="Plot the Natural Neural Tangent Kernel of the network every n steps [0 means never]",
                        type=int)

    parser.add_argument("-vb", "--verbosity",
                        help="Plot resutls on console every n steps [0 means never]",
                        type=int)

    parser.add_argument("-ns", "--nsteps",
                        help="Number of optimization steps",
                        type=int)

    parser.add_argument("--seed",
                        help="Seed to be used",
                        type=int)

    parser.add_argument("--rcond",
                        help="The rcond for the spectral cutoff in anagram [negative or null value means default value]",
                        type=float)

    parser.add_argument("-rabs", "--rcond_absolute",
                        help="If set, then rcond is taken as an absolute value and not relative to biggest singular value",
                        action="store_false")

    return parser


def parse(args, default_params):
    ap = Parameters(args.__dict__)
    dp = Parameters(default_params)
    correspondences = {
        'layer_sizes': 'layer_sizes',
        'expe_name': 'expe_name',
        'expe_path': 'path',
        'tensorboard_path': 'tensorboard_path',
        'save_final_weights': 'save_final_weights',
        'weights_verbosity': 'log_weights',
        'tb_verbosity': 'tensorboard',
        'NNTK_verbosity': 'NNTK_plot',
        'NTK_verbosity': 'NTK_plot',
        'Green_verbosity': 'Green_plot',
        'verbosity': 'verbosity',
        'nsteps': 'nsteps',
        'seed': 'seed',
        'rcond': 'rcond',
        'rcond_relative_to_bigger_sv': 'rcond_absolute',
    }

    if ap.layer_sizes is not None and not (dp.layer_sizes[0] == ap.layer_sizes[0] and dp.layer_sizes[-1] == ap.layer_sizes[-1]):
        raise ValueError(f'The chosen layer sizes does not correspond to problem. Input should have size {dp.layer_sizes[0]}'
                         f'instead of {ap.layer_sizes[0]}, and output {dp.layer_sizes[-1]} instead of {ap.layer_sizes[-1]}.')

    for keyp, keya in correspondences.items():
        if ap[keya] is not None:
            dp[keyp] = ap[keya]
    return dp

# extract samples from intergrators
def make_integrator_sample(integrator):
    return integrator._x

def make_integrators_samples(integrators):
    return tuple(make_integrator_sample(integrator) for integrator in integrators)

# Norms and losses
def l2_norm(f, x, params=None):
    return jnp.sqrt(l2_square_norm(f, x, params))

def generalized_sobolev_norm(f, generalized_sobolev_operator, samples, source=None):
    if source is None:
        return l2_norm(jax.vmap(generalized_sobolev_operator(f)), samples)
    operator_on_diff = jax.vmap(generalized_sobolev_operator(jax.jit(lambda x: f(x) - source(x))))
    return l2_norm(operator_on_diff, samples)
    
def generalized_sobolev_loss_factory(model, generalized_sobolev_operator, samples, source=None):
    diff = pre_quadratic_gradient_factory(model, identity_operator, source)
    sobolev_integrated = quadratic_gradient_factory(diff, generalized_sobolev_operator)
    return jax.jit(lambda params: l2_norm(sobolev_integrated, samples, params))

def l2_loss_factory(model, x, source=None):
    return generalized_sobolev_loss_factory(model, identity_operator, x, source)

def sobolev_operator(u):
    return jax.jit(lambda x: jnp.concatenate((u(x)[None], jax.grad(u)(x))))

def h1_norm(f, x, source=None):
    return generalized_sobolev_norm(f, sobolev_operator, x, source)

def h1_loss_factory(model, x, source=None):
    return generalized_sobolev_loss_factory(model, sobolev_operator, x, source)

class Assistant:
    def __init__(self,
                 integrators,
                 operators,
                 expe_parameters,
                 sources=None,
                 solution=None):
        if len(integrators) == len(operators): # in this case last integrator is interior integrator and used as eval integrator
            self.batch_samples = make_integrators_samples(integrators)
        elif len(integrators) == len(operators)+1: # in this case last integrator is eval integrator
            self.batch_samples = make_integrators_samples(integrators[:-1])
        else:
            raise ValueError('The number of operators should match the number of integrators')
        self.eval_samples = make_integrator_sample(integrators[-1])

        if sources is not None and not len(sources) == len(operators):
            raise ValueError('The number of operators should match the number of sources')

        self.ep = ep = Parameters(expe_parameters)
        self.solution = solution
        for key, val in ep.items():
            setattr(self, key, val)

        for key in {'verbosity', 'tb_verbosity', 'weights_verbosity', 'NNTK_verbosity', 'NTK_verbosity', 'Green_verbosity'}:
            setattr(self, key, max(ep[key], 0))

        test_expe_name_is_file = None if self.expe_name is None else re.match('^([^\.]*)(?:\.[a-zA-Z]*)$', self.expe_name)
        if test_expe_name_is_file is not None:
            self.expe_name = test_expe_name_is_file.group(1)

        if isinstance(operators, dict):
            self.operators_names = tuple(operators.keys())
            operators = tuple(operators.values())
        else:
            self.operators_names = tuple(op.__name__ for op in operators)

        self.model = mlp(jnp.tanh)
        key = jax.random.PRNGKey(self.seed)
        self.random_key, key_nn, key_np = jax.random.split(key, 3)

        self.rng_np = np.random.default_rng(np.asarray(key_np))
        self.init_params = init_params(self.layer_sizes, key_nn)

        if self.rcond is not None and self.rcond <= 0.:
            self.rcond = None

        self.save_final_weights = False if self.save_final_weights is None else True

        if self.save_final_weights or self.weights_verbosity or self.NNTK_verbosity or self.Green_verbosity or (self.tb_verbosity and self.tensorboard_path is None):
            if self.expe_path is None:
                if self.expe_name is None:
                    raise ValueError('You need to provide a path or an expe_name.')
                else:
                    self.expe_path = os.path.join('experiments', '{}_{}'.format(self.expe_name, datetime.now().strftime("%Y%m%d-%H%M%S")))
            try:
                os.makedirs(self.expe_path)
            except FileExistsError as e:
                e.strerror = 'The experiment path already exists'
                raise e

        if self.tb_verbosity:
            tensorboard_folder_name = datetime.now().strftime("%Y%m%d-%H%M%S") if self.expe_name is None \
                                        else '{}_{}'.format(self.expe_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
            if self.tensorboard_path is None:
                self.tensorboard_path = os.path.join('experiments', 'tensorboard_logs')
            self.tensorboard_path = os.path.join(self.tensorboard_path, tensorboard_folder_name)
            self.summary_writer = tfsum.create_file_writer(self.tensorboard_path)


        if self.weights_verbosity or self.save_final_weights:
            self.weights_path = os.path.join(self.expe_path, 'weights')
            os.makedirs(self.weights_path)

        if self.NNTK_verbosity:
            self.NNTK_path = os.path.join(self.expe_path, 'NNTK')
            os.makedirs(self.NNTK_path)
            self.full_features = full_features_factory(self.model, operators)
            self.main_operator_features = features_factory(self.model, operators[-1])

        if self.NTK_verbosity:
            self.NTK_path = os.path.join(self.expe_path, 'NTK')
            os.makedirs(self.NTK_path)
            try:
                self.full_features
            except AttributeError:
                self.full_features = full_features_factory(self.model, operators)
            self.main_operator_features = features_factory(self.model, operators[-1])

        if self.Green_verbosity:
            self.Green_path = os.path.join(self.expe_path, 'Green')
            os.makedirs(self.Green_path)
            try:
                self.full_features
            except AttributeError:
                self.full_features = full_features_factory(self.model, operators)
            self.model_features = features_factory(self.model, identity_operator)

        if solution is not None:
            self.L2_error = l2_loss_factory(self.model, self.eval_samples, solution)
            self.H1_error = h1_loss_factory(self.model, self.eval_samples, solution)
        else:
            self.L2_error = self.H1_error = None

        self.optimizer = Optimizer(self.model, make_integrators_samples(integrators[:-1]), operators, sources, self.rcond, self.rcond_relative_to_bigger_sv)

        self.start_time = time.time()

    def before_loop(self, optimizer, params, samples, n_steps):
        self.start_loop_time = time.time()
        elapsed_time = self.start_loop_time - self.start_time

        print(
            f'Time elapsed since beginning : {elapsed_time:.2f}s\n'
            f'Starting now optimization loop for {n_steps} steps, displaying metrics every {self.verbosity} steps\n'
            f'The seed used is {self.seed}\n'
        )

    def after_update(self, optimizer, params, samples, n_steps, iteration, actual_step, nat_grads):
        l2_error = None
        if self.verbosity and iteration % self.verbosity == 0:
            l2_error = self.L2_error(params)
            h1_error = self.H1_error(params)
            loss = optimizer.tot_loss(params)

            # Time calculation
            actual_time = time.time()
            loop_time = (actual_time - self.start_loop_time) / (iteration+1)
            elapsed_time = actual_time - self.start_time

            print(
                f'NGD Iteration: {iteration} with loss: '
                f'{loss}, error L2: {l2_error}, error H1: {h1_error} and step: {actual_step}\n'
                f'Total time elapsed: {elapsed_time:.2f}s Mean time per step : {loop_time:.2f}s'
            )

        if self.tb_verbosity and iteration % self.tb_verbosity == 0:
            if l2_error is None:
                l2_error = self.L2_error(params)
                h1_error = self.H1_error(params)
                elapsed_time = time.time() - self.start_time
                loss = optimizer.tot_loss(params)

            write_to_tensorboard(self.summary_writer, iteration, dict({
                'loss': loss,
                'l2_error': l2_error, 'h1_error': h1_error,
                'lr': actual_step, 'elapsed_time': elapsed_time,
            }, **{f'{op_name}_loss': op(params) for op_name, op in zip(self.operators_names, optimizer.losses)}))
            l2_error = None

        if self.NNTK_verbosity and iteration % self.NNTK_verbosity == 0:
            plot_NNTK(self.NNTK_path,
                      self.full_features,
                      self.main_operator_features,
                      iteration,
                      params,
                      samples,
                      self.eval_samples,
                      self.rcond)

        if self.NTK_verbosity and iteration % self.NTK_verbosity == 0:
            plot_NTK(self.NTK_path,
                     self.full_features,
                     self.main_operator_features,
                     iteration,
                     params,
                     samples,
                     self.eval_samples,
                     self.rcond)

        if self.Green_verbosity and iteration % self.Green_verbosity == 0:
            pass
            # plot_Green(self.Green_path,
            #            self.full_features,
            #            self.model_features,
            #            iteration,
            #            params,
            #            samples,
            #            self.eval_samples,
            #            self.rcond)

        if self.weights_verbosity and iteration % self.weights_verbosity == 0:
            write_weights(self.weights_path, iteration, params)

    def after_loop(self, optimizer, params, samples, n_steps):
        write_weights(self.weights_path, 'final', params)

    def optimize(self):
        self.optimizer.optimize(self.nsteps, self.init_params, self.batch_samples,
                                {
                                    'before_loop': self.before_loop,
                                    'after_update': self.after_update,
                                    'after_loop': self.after_loop,
                                 })
