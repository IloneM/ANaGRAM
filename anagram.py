#!/usr/bin/env python
# coding: utf-8

# In[0]:


"""
Implementation of the ANaGRAM and its natural gradients.

"""
import jax.numpy as jnp
from jax import grad, vmap, jit
import jax.flatten_util
import jax.scipy.linalg
from ngrad.utility import del_i
from functools import partial


# In[1]:


def identity_operator(u):
    return u

# In[2]:


ddxi = lambda u, i: del_i(del_i(u, i), i)

def laplacian(u, axis):
    full_double_grad = [ddxi(u, i) for i in axis]
    return jax.jit(lambda x: jnp.sum(jnp.stack([d(x) for d in full_double_grad])))

# In[3]:


def make_operator_on_model(model, functional_operator):
    return jit(lambda params, x: functional_operator(lambda z: model(params, z))(x))

# In[4]:


def pre_features_factory(model, functional_operator):
    operator_on_model = make_operator_on_model(model, functional_operator)

    @jit
    def del_theta_operator(params, x):
        return grad(operator_on_model, argnums=0)(params, x)

    @jit
    def pre_features(params, x):
        return jax.flatten_util.ravel_pytree(del_theta_operator(params, x))[0]

    return pre_features

def features_factory(model, functional_operator):
    pre_features = pre_features_factory(model, functional_operator)
    v_pre_features = vmap(pre_features, (None, 0))

    @jit
    def features(params, samples):
        return v_pre_features(params, samples).T
    
    return features


# In[5]:


def full_features_factory(model, functional_operators):
    features_by_operator = tuple(features_factory(model, fo) for fo in functional_operators)
    
    @jit
    def full_features(params, batch_samples):
        return jnp.concatenate(tuple(fbo(params, bs) for fbo, bs in zip(features_by_operator, batch_samples)), axis=1)

    return full_features
    

# In[6]:


def nat_grad_factory(full_features, true_gradient, const_tol=None, const_cut_low_signal=True, const_return_details=False):
    @partial(jax.jit, static_argnames=['tol', 'cut_low_signal', 'return_details'])
    def natural_gradient(params, batch_samples,
                         tol=const_tol, cut_low_signal=const_cut_low_signal, return_details=const_return_details):
        full_features_evaluated = full_features(params, batch_samples)
        
        U_features, Lambda_features, VT_features = jax.scipy.linalg.svd(full_features_evaluated, full_matrices=False)
        if tol is None:
            tol = jnp.sqrt(jnp.finfo(Lambda_features.dtype).eps * full_features_evaluated.shape[0]) * Lambda_features[0]
        mask = Lambda_features >= jnp.array(tol, dtype=Lambda_features.dtype)
        rank = mask.sum()
        safe_Lambda = jnp.where(mask, Lambda_features, 1).astype(Lambda_features.dtype)
        Lambda_inv = jnp.where(mask, 1 / safe_Lambda, 0)
        
        true_gradient_evaluated = true_gradient(params, batch_samples)
        retriev_pytree  = jax.flatten_util.ravel_pytree(params)[1]

        gradient_rotated = VT_features @ true_gradient_evaluated
        
        flat_nat_grad = flat_nat_grad_cutted = U_features @ (Lambda_inv * gradient_rotated)
        if not cut_low_signal:
            low_signal_projection = true_gradient_evaluated - VT_features.T @ gradient_rotated
            flat_nat_grad = flat_nat_grad + low_signal_projection

        if return_details:
            residual = true_gradient_evaluated - full_features_evaluated.T @ flat_nat_grad_cutted
            return retriev_pytree(flat_nat_grad), residual, rank, Lambda_features
        else:
            return retriev_pytree(flat_nat_grad)

    return natural_gradient

# In[7]:

def null_source(x):
    return 0.

# In[8]:

def pre_quadratic_gradient_factory(model, functional_operator, source=None):
    operator_on_model = make_operator_on_model(model, functional_operator)
    if source is None:
        source = null_source
    return jax.jit(lambda params, x: operator_on_model(params, x) - jax.jit(source)(x))

# In[9]:

def quadratic_gradient_factory(model, functional_operator, source=None):
    return vmap(pre_quadratic_gradient_factory(model, functional_operator, source), (None, 0))

# In[10]:

def full_quadratic_gradient_factory(model, functional_operators, sources=None):
    if sources is None:
        sources = tuple(None for fo in functional_operators)
    simple_gradients = tuple(quadratic_gradient_factory(model, fo, so) for fo, so in zip(functional_operators, sources))

    @jit
    def full_gradient(params, batch_samples):
        return jnp.concatenate(tuple(sg(params, bs) for sg, bs in zip(simple_gradients, batch_samples)), axis=0)

    return full_gradient

# def full_quadratic_gradient_factory(model, functional_operators, sources=None):
#     operators_on_model = tuple(make_operator_on_model(model, fo) for fo in functional_operators)
#     v_operators_on_model = tuple(vmap(oom, (None, 0)) for oom in operators_on_model)

#     if sources is None:
#         sources = tuple(null_source for fo in functional_operators)
#     v_sources = tuple(vmap(s, (0,)) for s in sources)

#     @jit
#     def full_gradient(params, batch_samples):
#         return jnp.concatenate(tuple(voom(params, bs) - vs(bs)
#                                      for voom, vs, bs in zip(v_operators_on_model, v_sources, batch_samples)), axis=0)

#     return full_gradient

# In[11]:

# def quadratic_nat_grad_factory(model, functional_operators, const_tol=None, sources=None, const_cut_low_signal=True, const_return_details=False):
def quadratic_nat_grad_factory(model, functional_operators, sources=None, tol=None, cut_low_signal=True, return_details=False):
    full_features = full_features_factory(model, functional_operators)
    true_gradient = full_quadratic_gradient_factory(model, functional_operators, sources)

    return nat_grad_factory(full_features, true_gradient, tol, cut_low_signal, return_details)

