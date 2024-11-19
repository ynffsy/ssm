from tqdm.auto import trange
from scipy.optimize import linear_sum_assignment

import jax.numpy as np
from jax import random
from jax.nn import one_hot as jax_one_hot



SEED = hash("ssm") % (2**32)
LOG_EPS = 1e-16
DIV_EPS = 1e-16



def compute_state_overlap(z1, z2, K1=None, K2=None):
    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    # Use one-hot encoding and matrix multiplication to compute overlap
    z1_one_hot = jax_one_hot(z1, K1)
    z2_one_hot = jax_one_hot(z2, K2)

    overlap = np.einsum('ti,tj->ij', z1_one_hot, z2_one_hot)
    return overlap


def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm


def rle(stateseq):
    """
    Compute the run length encoding of a discrete state sequence.

    E.g., the state sequence [0, 0, 1, 1, 1, 2, 3, 3]
    would be encoded as ([0, 1, 2, 3], [2, 3, 1, 2])

    Parameters
    ----------
    stateseq : array_like
        Discrete state sequence.

    Returns
    -------
    ids : array_like
        Integer identities of the states.

    durations : array_like (int)
        Length of time in corresponding state.
    """
    pos = np.where(np.diff(stateseq) != 0)[0]
    pos = np.concatenate((np.array([0]), pos + 1, np.array([len(stateseq)])))
    return stateseq[pos[:-1]], np.diff(pos)


def random_rotation(key, n, theta=None):
    if theta is None:
        key, subkey = random.split(key)
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * random.uniform(subkey)

    if n == 1:
        key, subkey = random.split(key)
        rotation_matrix = random.uniform(subkey) * np.eye(1)
        return rotation_matrix, key

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    out = np.eye(n)
    out = out.at[:2, :2].set(rot)

    key, subkey = random.split(key)
    q, _ = np.linalg.qr(random.normal(subkey, (n, n)))
    rotation_matrix = q @ out @ q.T
    return rotation_matrix, key


def ensure_args_are_lists(f):
    def wrapper(self, datas, inputs=None, masks=None, tags=None, **kwargs):
        datas = [datas] if not isinstance(datas, (list, tuple)) else datas

        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(M, tuple)

        if inputs is None:
            inputs = [np.zeros((data.shape[0],) + M) for data in datas]
        elif not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if masks is None:
            masks = [np.ones_like(data, dtype=bool) for data in datas]
        elif not isinstance(masks, (list, tuple)):
            masks = [masks]

        if tags is None:
            tags = [None] * len(datas)
        elif not isinstance(tags, (list, tuple)):
            tags = [tags]

        return f(self, datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

    return wrapper


def ensure_variational_args_are_lists(f):
    def wrapper(self, arg0, datas, inputs=None, masks=None, tags=None, **kwargs):
        datas = [datas] if not isinstance(datas, (list, tuple)) else datas

        try:
            M = (self.M,) if isinstance(self.M, int) else self.M
        except AttributeError:
            M = (arg0.M,) if isinstance(arg0.M, int) else arg0.M

        assert isinstance(M, tuple)

        if inputs is None:
            inputs = [np.zeros((data.shape[0],) + M) for data in datas]
        elif not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if masks is None:
            masks = [np.ones_like(data, dtype=bool) for data in datas]
        elif not isinstance(masks, (list, tuple)):
            masks = [masks]

        if tags is None:
            tags = [None] * len(datas)
        elif not isinstance(tags, (list, tuple)):
            tags = [tags]

        return f(self, arg0, datas, inputs=inputs, masks=masks, tags=tags, **kwargs)

    return wrapper


def ensure_args_not_none(f):
    def wrapper(self, data, input=None, mask=None, tag=None, **kwargs):
        assert data is not None

        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(M, tuple)
        input = np.zeros((data.shape[0],) + M) if input is None else input

        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return f(self, data, input=input, mask=mask, tag=tag, **kwargs)
    return wrapper


def ensure_slds_args_not_none(f):
    def wrapper(self, variational_mean, data, input=None, mask=None, tag=None, **kwargs):
        assert variational_mean is not None
        assert data is not None
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(M, tuple)
        input = np.zeros((data.shape[0],) + M) if input is None else input
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return f(self, variational_mean, data, input=input, mask=mask, tag=tag, **kwargs)
    return wrapper


def ssm_pbar(num_iters, verbose, description, prob):
    """
    Return either progress bar or regular list for iterating.

    Parameters
    ----------
    num_iters : int
        Number of iterations.

    verbose : int
        If == 2, return trange object; else returns list.

    description : str
        Description for progress bar.

    prob : float
        Values to initialize description fields at.
    """
    if verbose == 2:
        pbar = trange(num_iters)
        pbar.set_description(description.format(*prob))
    else:
        pbar = range(num_iters)
    return pbar


def logistic(x):
    return 1. / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def softplus(x):
    return np.log1p(np.exp(x))


def inv_softplus(y):
    return np.log(np.exp(y) - 1)


def one_hot(z, K):
    return jax_one_hot(z, K)


def relu(x):
    return np.maximum(0, x)


def replicate(x, state_map, axis=-1):
    """
    Replicate an array of shape (..., K) according to the given state map
    to get an array of shape (..., R) where R is the total number of states.

    Parameters
    ----------
    x : array_like, shape (..., K)
        The array to be replicated.

    state_map : array_like, shape (R,), int
        The mapping from [0, K) -> [0, R).
    """
    assert state_map.ndim == 1
    assert np.all(state_map >= 0) and np.all(state_map < x.shape[-1])
    return np.take(x, state_map, axis=axis)


def collapse(x, state_map, axis=-1):
    """
    Collapse an array of shape (..., R) to shape (..., K) by summing
    columns that map to the same state in [0, K).

    Parameters
    ----------
    x : array_like, shape (..., R)
        The array to be collapsed.

    state_map : array_like, shape (R,), int
        The mapping from [0, K) -> [0, R).
    """
    R = x.shape[axis]
    assert state_map.ndim == 1 and state_map.shape[0] == R
    K = state_map.max() + 1

    def sum_over_axis(k):
        indices = np.where(state_map == k)[0]
        return np.sum(np.take(x, indices, axis=axis), axis=axis, keepdims=True)

    collapsed = np.concatenate([sum_over_axis(k) for k in range(K)], axis=axis)
    return collapsed


def check_shape(var, var_name, desired_shape):
    assert var.shape == desired_shape, f"Variable {var_name} is of wrong shape. Expected {desired_shape}, found {var.shape}."


def trace_product(A, B):
    """
    Compute trace of the matrix product A*B efficiently.

    A, B can be 2D or 3D arrays, in which case the trace is computed along
    the last two axes. In this case, the function will return an array.
    Computed using the fact that tr(AB) = sum_{ij} A_{ij} B_{ji}.
    """
    ndimsA = A.ndim
    ndimsB = B.ndim
    assert ndimsA == ndimsB, "Both A and B must have the same number of dimensions."
    assert ndimsA <= 3, "A and B must have 3 or fewer dimensions."

    # Take the trace along the last two dimensions.
    BT = np.swapaxes(B, -1, -2)
    return np.sum(A * BT, axis=(-1, -2))
