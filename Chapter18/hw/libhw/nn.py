"""
Simple set of NN operations
"""
import math as m


def matmul(a, b):
    """
    Multipy two matrices
    :param a: list of list of floats
    :param b: list of list of floats
    :return: resulting matrix
    """
    res = []
    for r_idx, a_r in enumerate(a):
        res_row = [
            sum([a_v * b_r[b_idx] for a_v, b_r in zip(a_r, b)])
            for b_idx in range(len(b[0]))
        ]
        res.append(res_row)
    return res


def matadd_t(a, b):
    """
    Inplace add of transposed b matrix
    """
    for a_idx, r_a in enumerate(a):
        for idx in range(len(r_a)):
            r_a[idx] += b[idx][a_idx]
    return a


def apply(m, f):
    """
    Apply operation in-place
    :param m:
    :param f:
    :return:
    """
    for r in m:
        for idx, v in enumerate(r):
            r[idx] = f(v)
    return m


def relu(x):
    return apply(x, lambda v: 0.0 if v < 0.0 else v)


def tanh(x):
    return apply(x, m.tanh)


def linear(x, w_pair):
    w, b = w_pair
    return matadd_t(matmul(w, x), b)
