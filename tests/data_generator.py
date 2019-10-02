import numpy as np
import pandas as pd


def _get_x(vec_t, num_col):
    mat = np.ones((len(vec_t), num_col))
    for i in range(1, num_col):
        mat[:, i] = np.power(vec_t, i)
    return mat


def _sim_response(bta, lmd, gma, vec_n, vec_t, num_col):
    """
    Simulate the responses from Normal distributions
    """
    vec_y = np.empty(0)
    # set indices
    first, last = 0, 0
    for ni in vec_n:
        last = first + ni
        # get a sub-vector of covariates for the i-th subject
        ti = vec_t[first:last]
        # get design matrices
        xi = _get_x(ti, num_col[0])
        zi = _get_x(ti, num_col[1])
        # get mean
        mu_i = xi @ bta
        sigma2_i = np.exp(zi @ lmd)
        # initialise yi with mui
        yi = mu_i

        if ni == 1:
            yi += np.random.normal(0., np.sqrt(sigma2_i[0]))
        else:
            for j in range(1, ni):
                for k in range(j):
                    # get w
                    wijk = [(ti[j] - ti[k])**l for l in range(num_col[2])]
                    # get phi
                    phi_ijk = np.array(wijk) @ gma
                    # update yi
                    yi[j] += phi_ijk * (yi[k] - mu_i[k])
                # add random term
                yi[j] += np.random.normal(0., np.sqrt(sigma2_i[j]))
        vec_y = np.append(vec_y, yi)
        first = last
    return vec_y


def _get_id(vec_n):
    """
    Get the ID numbers for each measurement
    """
    vec_id = np.empty(0)
    for i in range(len(vec_n)):
        for j in range(vec_n[i]):
            vec_id = np.append(vec_id, i)
    return vec_id


def _sim_t(vec_n):
    """
    Simulate time from uniform distribution
    """
    vec_t = np.empty(0)
    for ni in vec_n:
        ti = np.random.uniform(-1, 1, ni)
        vec_t = np.append(vec_t, np.sort(ti))
    return vec_t


def generator(m=10):
    """
    Generate the data given the model parameters
    -m: Number of subject simulated
    """
    # simulate the number of parameters
    n_bta, n_lmd, n_gma = np.random.randint(1, 5, (3,))

    # simulate the paramters
    bta = np.random.uniform(-5, 5, n_bta)
    lmd = np.random.uniform(-5, 5, n_lmd)
    gma = np.random.uniform(-5, 5, n_gma)
    # number of measurements for each subject
    vec_n = np.random.randint(1, 10, (m,))
    vec_id = _get_id(vec_n)
    # generate the vector of time
    vec_t = _sim_t(vec_n)

    num_col = (n_bta, n_lmd, n_gma)
    vec_y = _sim_response(bta, lmd, gma, vec_n, vec_t, num_col)

    df = pd.DataFrame({'y': vec_y, 'id': vec_id, 't': vec_t})
    return df

