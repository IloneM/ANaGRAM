#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Implementation of the ANaGRAM and its natural gradients.

"""
import jax.numpy as jnp
from jax import grad, vmap, jit
import jax.flatten_util
import jax.scipy.linalg


# In[2]:


def identity_operator(u):
    return u

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
        features_matrix = v_pre_features(params, samples)
        return features_matrix
    
    return features


# In[5]:


def full_features_factory(model, functional_operators):
    features_by_operator = tuple(features_factory(model, fo) for fo in functional_operators)
    
    @jit
    def full_features(params, batch_samples):
        return jnp.concatenate(tuple(fbo(params, bs).T for fbo, bs in zip(features_by_operator, batch_samples)), axis=1)

    return full_features


# In[6]:


def nat_grad_factory(full_features, true_gradient, const_tol=None, const_cut_low_signal=True):
    def natural_gradient(params, batch_samples, tol=const_tol, cut_low_signal=const_cut_low_signal):
        if tol is None:
            raise ValueError('tol should be specified')
        full_features_evaluated = full_features(params, batch_samples)
        
        U_features, Lambda_features, VT_features = jax.scipy.linalg.svd(full_features_evaluated, full_matrices=False)
        tol_selector = Lambda_features > tol
        U_features, Lambda_features, VT_features = U_features[:, tol_selector], Lambda_features[tol_selector], VT_features[tol_selector]

        true_gradient_evaluated = true_gradient(params, batch_samples)
        
        retriev_pytree  = jax.flatten_util.ravel_pytree(params)[1]
        
        gradient_rotated = VT_features @ true_gradient_evaluated
        
        flat_nat_grad = U_features @ (gradient_rotated / Lambda_features)
        if not cut_low_signal:
            low_signal_projection = true_gradient_evaluated - VT_features.T @ gradient_rotated
            flat_nat_grad = flat_nat_grad + low_signal_projection

        return retriev_pytree(flat_nat_grad)

    return natural_gradient

# In[7]:

def null_source(x):
    return 0.


# In[8]:


def quadratic_gradient_factory(model, functional_operators, sources=None):
    operators_on_model = tuple(make_operator_on_model(model, fo) for fo in functional_operators)
    v_operators_on_model = tuple(vmap(oom, (None, 0)) for oom in operators_on_model)

    if sources is None:
        sources = tuple(null_source for fo in functional_operators)
    v_sources = tuple(vmap(s, (0,)) for s in sources)

    @jit
    def full_gradient(params, batch_samples):
        return jnp.concatenate(tuple(voom(params, bs) - vs(bs)
                                     for voom, vs, bs in zip(v_operators_on_model, v_sources, batch_samples)), axis=0)

    return full_gradient


# In[9]:


def quadratic_nat_grad_factory(model, functional_operators, const_tol, sources=None, const_cut_low_signal=True):
    full_features = full_features_factory(model, functional_operators)
    true_gradient = quadratic_gradient_factory(model, functional_operators, sources)

    return nat_grad_factory(full_features, true_gradient, const_tol, const_cut_low_signal)

