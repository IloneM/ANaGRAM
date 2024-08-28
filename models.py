import jax.numpy as jnp
import jax
import optax

from jax.typing import ArrayLike, DTypeLike
from typing import Sequence, Callable, Tuple

Activation = Callable[[ArrayLike], jax.Array]
Model = Callable[[optax.Params, ArrayLike], jax.Array]

def glorot_normal(key: ArrayLike, shape: Sequence[int], dtype: DTypeLike = float) -> jax.Array:
    assert len(shape) == 2
    factor = jnp.sqrt(jnp.array(2. / sum(shape)))
    return (jax.random.normal(key, shape, dtype) * factor).astype(dtype)


def init_params(layer_sizes: Sequence[int], key: ArrayLike, dtype: DTypeLike = float) -> optax.Params:
    lastsize = layer_sizes[0]
    initial_params = [None] * (len(layer_sizes) - 1)
    totsize = 0

    nn_keys = jax.random.split(key, len(layer_sizes) - 1)

    for i, (layersize, nn_key) in enumerate(zip(layer_sizes[1:], nn_keys)):
        initial_params[i] = (glorot_normal(nn_key, (lastsize, layersize), dtype), jnp.zeros(layersize, dtype=dtype))

        totsize += (lastsize + 1) * layersize
        lastsize = layersize

    print(f'Creating array of {totsize} parameters')

    return initial_params

def mlp(activations: Activation | Sequence[Activation]) -> Tuple[Model, Model]:
    if callable(activations):
        @jax.jit
        def mlp_model(params: optax.Params, x: ArrayLike):
            for param in params:
                x = activations(jnp.dot(x, param[0]) + param[1])
            return jnp.reshape(x, ())
    else:
        @jax.jit
        def mlp_model(params: optax.Params, x: ArrayLike):
            for i, activation in enumerate(activations):
                x = activation(jnp.dot(x, params[i][0]) + params[i][1])
            return jnp.reshape(x, ())
    return mlp_model, jax.vmap(mlp_model, (None, 0))
