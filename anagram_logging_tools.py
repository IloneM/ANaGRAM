from tensorflow import summary as tfsum
import os
from typing import Iterable, Callable
from matplotlib import colors
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

# # Specify the class in __all__
# __all__ = ['Parser']

def plot_im(im_to_plot, fig=None, logarithmic=False, labels=('t', 'x')):
    resolution = int(jnp.sqrt(im_to_plot.shape[0]))
    extent = (0., 1., 1., 0.)

    close_fig = False
    if fig is None:
        fig = plt.figure(figsize=(10,10), dpi=100)
        close_fig = True

    fig.tight_layout()
    
    plt.imshow(im_to_plot.reshape(resolution, resolution), aspect='auto', extent=extent, norm=(colors.LogNorm() if logarithmic else None))
    plt.ylabel('${}$'.format(labels[0]))
    plt.xlabel('${}$'.format(labels[1]))
    plt.colorbar()
    if close_fig:
        plt.close()

def plot_random_kernels(K: jax.typing.ArrayLike,
                        batch: jax.typing.ArrayLike,
                        path: str,
                        K_name: str, iteration: int,
                        n_features: int = 16,
                        logarithmic: bool = False):
    n_features = min(K.shape[0], n_features)
    sqrt_n_features = int(np.ceil(np.sqrt(n_features)))
    n_features = sqrt_n_features ** 2
    
    fig = plt.figure(figsize=(n_features,n_features), dpi=100)
    
    plt.suptitle('{} random selected {} for iteration {}'.format(n_features, K_name, iteration))
    for i, e in enumerate(np.random.choice(K.shape[0], n_features, replace=None)):
        plt.subplot(sqrt_n_features,sqrt_n_features,i+1)
        plot_im(K[e], fig, logarithmic=logarithmic)
        plt.scatter(*batch[e][::-1], marker="+", c='red')
    plt.savefig(os.path.join(path, '{}_{}.png'.format(K_name, iteration)))
    plt.close()

def compute_NNTK(features_ref: Callable[[Iterable[jax.typing.ArrayLike], Iterable[jax.typing.ArrayLike]], jax.Array],
                 features_computing: Callable[[Iterable[jax.typing.ArrayLike], jax.typing.ArrayLike], jax.Array],
                 batch_ref: Iterable[jax.typing.ArrayLike],
                 batch_computing: jax.typing.ArrayLike,
                 params: Iterable[jax.typing.ArrayLike],
                 tol: int|None = None):
    features_eval_ref = features_ref(params, batch_ref)
    features_eval_computing = features_computing(params, batch_computing)

    U_features, Lambda_features, VT_features = jax.scipy.linalg.svd(features_eval_ref, full_matrices=False)
    if tol is None:
        tol = jnp.sqrt(jnp.finfo(Lambda_features.dtype).eps * full_features_evaluated.shape[0]) * Lambda_features[0]
    mask = Lambda_features >= jnp.array(tol, dtype=Lambda_features.dtype)
    # rank = mask.sum()
    safe_Lambda = jnp.where(mask, Lambda_features, 1).astype(Lambda_features.dtype)
    Lambda_inv = jnp.where(mask, 1 / safe_Lambda, 0)[:,jnp.newaxis]
    return VT_features.T @ ((U_features.T @ features_eval_computing) * Lambda_inv)

def compute_NTK(features_ref: Callable[[Iterable[jax.typing.ArrayLike], Iterable[jax.typing.ArrayLike]], jax.Array],
                features_computing: Callable[[Iterable[jax.typing.ArrayLike], jax.typing.ArrayLike], jax.Array],
                batch_ref: Iterable[jax.typing.ArrayLike],
                batch_computing: jax.typing.ArrayLike,
                params: Iterable[jax.typing.ArrayLike]):
    features_eval_ref = features_ref(params, batch_ref)
    features_eval_computing = features_computing(params, batch_computing)
    return features_eval_ref.T @ features_eval_computing

# def compute_Green(features_ref: Callable[[Iterable[jax.typing.ArrayLike], Iterable[jax.typing.ArrayLike]], jax.Array],
#                   features_computing: Callable[[Iterable[jax.typing.ArrayLike], jax.typing.ArrayLike], jax.Array],
#                   batch_ref: Iterable[jax.typing.ArrayLike],
#                   batch_computing: jax.typing.ArrayLike,
#                   params: Iterable[jax.typing.ArrayLike],
#                   tol: int|None = None):
#     features_eval_ref = features_ref(params, batch_ref)
#     features_eval_computing = features_computing(params, batch_computing)

#     U_features, Lambda_features, VT_features = jax.scipy.linalg.svd(features_eval_ref, full_matrices=False)
#     if tol is None:
#         tol = jnp.sqrt(jnp.finfo(Lambda_features.dtype).eps * full_features_evaluated.shape[0]) * Lambda_features[0]
#     mask = Lambda_features >= jnp.array(tol, dtype=Lambda_features.dtype)
#     # rank = mask.sum()
#     safe_Lambda = jnp.where(mask, Lambda_features, 1).astype(Lambda_features.dtype)
#     Lambda_inv = jnp.where(mask, 1 / safe_Lambda, 0)[:,jnp.newaxis]
#     return VT_features.T @ ((U_features.T @ features_eval_computing) * Lambda_inv)

def plot_NNTK(path: str,
              features_ref: Callable[[Iterable[jax.typing.ArrayLike], Iterable[jax.typing.ArrayLike]], jax.Array],
              features_computing: Callable[[Iterable[jax.typing.ArrayLike], jax.typing.ArrayLike], jax.Array],
              iteration: int,
              params: Iterable[jax.typing.ArrayLike],
              batch_samples: Iterable[jax.typing.ArrayLike],
              computing_samples: jax.typing.ArrayLike|None = None,
              tol: float|None = None,
              n_features: int = 16,
              loarithmic: bool = True):
    if computing_samples is None:
        computing_samples = batch_samples[-1]
    NNTK = compute_NNTK(features_ref, features_computing, batch_samples, computing_samples, params, tol)
    plot_random_kernels(NNTK, jnp.concatenate(batch_samples, axis=0),
                        path, 'NNTK', iteration, n_features, loarithmic)

def plot_NTK(path: str,
             features_ref: Callable[[Iterable[jax.typing.ArrayLike], Iterable[jax.typing.ArrayLike]], jax.Array],
             features_computing: Callable[[Iterable[jax.typing.ArrayLike], jax.typing.ArrayLike], jax.Array],
             iteration: int,
             params: Iterable[jax.typing.ArrayLike],
             batch_samples: Iterable[jax.typing.ArrayLike],
             computing_samples: jax.typing.ArrayLike|None = None,
             tol: float|None = None,
             n_features: int = 16,
             loarithmic: bool = False):
    if computing_samples is None:
        computing_samples = batch_samples[-1]
    NTK = compute_NTK(features_ref, features_computing, batch_samples, computing_samples, params)
    plot_random_kernels(NTK, jnp.concatenate(batch_samples, axis=0),
                        path, 'NTK', iteration, n_features, loarithmic)

# def plot_Green(path: str,
#                features_ref: Callable[[Iterable[jax.typing.ArrayLike], Iterable[jax.typing.ArrayLike]], jax.Array],
#                features_computing: Callable[[Iterable[jax.typing.ArrayLike], jax.typing.ArrayLike], jax.Array],
#                iteration: int,
#                params: Iterable[jax.typing.ArrayLike],
#                batch_samples: Iterable[jax.typing.ArrayLike],
#                computing_samples: jax.typing.ArrayLike|None = None,
#                tol: float|None = None,
#                n_features: int = 16,
#                loarithmic: bool = True):
#     if computing_samples is None:
#         computing_samples = batch_samples[-1]
#     Green = compute_Green(features_ref, features_computing, batch_samples, computing_samples, params, tol)
#     plot_random_kernels(Green, jnp.concatenate(batch_samples, axis=0),
#                         path, 'Green kernel', iteration, n_features, loarithmic)

def write_weights(path: str, iteration: int|str, params: Iterable[jax.typing.ArrayLike]):
    jax.numpy.savez(os.path.join(path, 'weights_{}'.format(iteration)), *jax.tree_util.tree_leaves(params))


def write_to_tensorboard(output: str|tfsum.SummaryWriter, iteration: int, metrics: dict[str, jax.typing.ArrayLike]):
    summary_writer = tfsum.create_file_writer(output) if isinstance(output, str) else output
    with summary_writer.as_default():
        for name, value in metrics.items():
            tfsum.scalar(name, value, step=iteration)
