import copy
import warnings

import jax.numpy as np
import jax.random as random
from jax import grad, jit, value_and_grad
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
import optax  # For optimizers

from ssm.util import ensure_args_are_lists  # Ensure this utility is compatible with JAX

class _Hierarchical(object):
    """
    Base class for hierarchical models. Maintains a parent class and a
    bunch of children with their own perturbed parameters.
    """

    def __init__(self, base_class, *args, tags=(None,), lmbda=0.01, key=None, **kwargs):
        # Variance of child params around parent params
        self.lmbda = lmbda

        if key is None:
            raise ValueError("A PRNG key must be provided")
        self.key = key

        # Top-level parameters (parent)
        self.parent = base_class(*args, **kwargs)

        # Make models for each tag
        self.tags = tags
        self.children = dict()
        for tag in self.tags:
            self.key, subkey = random.split(self.key)
            ch = self.children[tag] = base_class(*args, **kwargs)
            ch_params = []
            for prm in self.parent.params:
                subkey, subsubkey = random.split(subkey)
                noise = np.sqrt(lmbda) * random.normal(subsubkey, shape=prm.shape)
                ch_params.append(prm + noise)
            ch.params = tuple(ch_params)

    @property
    def params(self):
        prms = (self.parent.params,)
        for tag in self.tags:
            prms += (self.children[tag].params,)
        return prms

    @params.setter
    def params(self, value):
        self.parent.params = value[0]
        for tag, prms in zip(self.tags, value[1:]):
            self.children[tag].params = prms

    def permute(self, perm):
        self.parent.permute(perm)
        for tag in self.tags:
            self.children[tag].permute(perm)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        self.parent.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        for tag in self.tags:
            self.children[tag].params = copy.deepcopy(self.parent.params)

    def log_prior(self, params):
        parent_params = params[0]
        child_params_list = params[1:]

        lp = self.parent.log_prior(parent_params)

        # Gaussian likelihood on each child param given parent param
        for child_params in child_params_list:
            for pprm, cprm in zip(parent_params, child_params):
                lp += np.sum(norm.logpdf(cprm, pprm, self.lmbda))
        return lp

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer_name="adam", num_iters=25, **kwargs):
        for tag in tags:
            if tag not in self.tags:
                raise Exception("Invalid tag: {}".format(tag))

        # Prepare the optimizer
        learning_rate = kwargs.get('learning_rate', 1e-3)
        if optimizer_name == "adam":
            opt = optax.adam(learning_rate)
        elif optimizer_name == "sgd":
            opt = optax.sgd(learning_rate)
        else:
            raise ValueError("Unknown optimizer {}".format(optimizer_name))

        # Expected log joint function
        def _expected_log_joint(params, expectations):
            parent_params = params[0]
            child_params_list = params[1:]

            # Compute log prior
            lp = self.parent.log_prior(parent_params)

            # Gaussian likelihood on each child param given parent param
            for child_params in child_params_list:
                for pprm, cprm in zip(parent_params, child_params):
                    lp += np.sum(norm.logpdf(cprm, pprm, self.lmbda))

            # Expected log likelihood
            elbo = lp
            for data, input, mask, tag, (expected_states, expected_joints), child_params \
                    in zip(datas, inputs, masks, tags, expectations, child_params_list):

                child = self.children[tag]
                # Temporarily set parameters for pure functions
                child_params_backup = child.params
                child.params = child_params

                if hasattr(child, 'log_initial_state_distn'):
                    log_pi0 = child.log_initial_state_distn(data, input, mask, tag)
                    elbo += np.sum(expected_states[0] * log_pi0)

                if hasattr(child, 'log_transition_matrices'):
                    log_Ps = child.log_transition_matrices(data, input, mask, tag)
                    elbo += np.sum(expected_joints * log_Ps)

                if hasattr(child, 'log_likelihoods'):
                    lls = child.log_likelihoods(data, input, mask, tag)
                    elbo += np.sum(expected_states * lls)

                # Restore original parameters
                child.params = child_params_backup

            return elbo

        # Define optimization target
        T = sum([data.shape[0] for data in datas])

        def _objective(params):
            obj = _expected_log_joint(params, expectations)
            return -obj / T

        # Compile the objective function for efficiency
        _objective = jit(_objective)
        value_and_grad_fn = value_and_grad(_objective)

        params = self.params
        opt_state = opt.init(params)

        for itr in range(num_iters):
            value, grads = value_and_grad_fn(params)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            # Optionally print progress or track loss
            # print(f"Iteration {itr}, Loss: {value}")

        self.params = params

class HierarchicalInitialStateDistribution(_Hierarchical):
    def log_initial_state_distn(self, data, input, mask, tag):
        log_pi0 = self.log_pi0 - logsumexp(self.log_pi0)
        return log_pi0

class HierarchicalTransitions(_Hierarchical):
    def log_transition_matrices(self, data, input, mask, tag):
        return self.children[tag].log_transition_matrices(data, input, mask, tag)

class HierarchicalObservations(_Hierarchical):
    def log_likelihoods(self, data, input, mask, tag):
        return self.children[tag].log_likelihoods(data, input, mask, tag)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        return self.children[tag].sample_x(z, xhist, input=input, tag=tag, with_noise=with_noise)

    def smooth(self, expectations, data, input, tag):
        return self.children[tag].smooth(expectations, data, input, tag)

class HierarchicalEmissions(_Hierarchical):
    def log_likelihoods(self, data, input, mask, tag):
        return self.children[tag].log_likelihoods(data, input, mask, tag)

    def sample_y(self, z, x, input=None, tag=None):
        return self.children[tag].sample_y(z, x, input=input, tag=tag)

    def initialize_variational_params(self, data, input, mask, tag):
        return self.children[tag].initialize_variational_params(data, input, mask, tag)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        return self.children[tag].smooth(expected_states, variational_mean, data, input, mask, tag)
