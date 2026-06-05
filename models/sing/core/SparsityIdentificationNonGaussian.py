"""
==============================================================================
Original Author: Ricardo Baptista (University of Toronto)
Repository: https://github.com/baptistar/SING/tree/73e1c2ad9a8886854c2ad22c75508ad3cc36df44/SING
Modified and Integrated by: Hyeok Kyu Kwon & Myeonggu Kang (2026)
==============================================================================
"""
import itertools
import pandas as pd
import numpy as np

from TransportMaps.Distributions import \
    StandardNormalDistribution, DistributionFromSamples, \
    PushForwardTransportMapDistribution, PullBackTransportMapDistribution
import models.sing.core.GeneralizedPrecision as GP
from TransportMaps import Default_IsotropicIntegratedSquaredTriangularTransportMap

__all__ = ['SING']

def SING(data, p_order, ordering, delta, offset=0, REG=None, plotting=False, delta_str=None):
    dim = data.shape[1]
    n_samps = data.shape[0]
    nax = np.newaxis
    pi = {}

    eta = StandardNormalDistribution(dim)
    qtype = 0
    qparams = n_samps
    ders = 2
    tol = 1e-5
    log = []

    active_vars = None
    sparsity = [0]
    sparsityIncreasing = True
    n_active_vars = [(np.power(dim, 2) + dim) / 2]

    perm_list = []
    total_perm = np.arange(dim)
    counter = 0

    results_list = []

    while sparsityIncreasing:
        pi = DistributionFromSamples(data)

        tm_approx = Default_IsotropicIntegratedSquaredTriangularTransportMap(
            dim, p_order, active_vars=active_vars, btype='fun')

        tm_density = PushForwardTransportMapDistribution(tm_approx, pi)
        solve = tm_density.minimize_kl_divergence(eta, qtype=qtype,
                                                  qparams=qparams,
                                                  regularization=REG,
                                                  tol=tol, ders=ders)

        pb_density = PullBackTransportMapDistribution(tm_approx, eta)

        ll_vec = pb_density.log_pdf(data)
        ll_mean = np.mean(ll_vec)

        omegaHat = GP.gen_precision(pb_density, data)

        gp_var = GP.var_omega(pb_density, data)
        tau = delta * np.sqrt(gp_var) * np.sqrt(np.log(n_samps))

        inv_perm_current = [0] * dim
        for idx, p in enumerate(total_perm):
            inv_perm_current[p] = idx

        omegaHat_pre_orig = omegaHat[:, inv_perm_current][inv_perm_current, :].copy()
        tau_orig = tau[:, inv_perm_current][inv_perm_current, :].copy()

        omegaHat_diagonal = np.copy(np.diag(omegaHat))
        omegaHat[np.abs(omegaHat) < offset + tau] = 0
        omegaHat[np.diag_indices_from(omegaHat)] = omegaHat_diagonal

        graph_post = np.zeros((dim, dim))
        graph_post[np.nonzero(omegaHat)] = 1
        graph_post_orig = graph_post[:, inv_perm_current][inv_perm_current, :].copy()

        results_list.append({
            'Iteration': counter + 1,
            'Omega_pre': omegaHat_pre_orig,
            'Tau': tau_orig,
            'Graph_post': graph_post_orig,
            'LogLikMean': ll_mean
        })

        perm1 = ordering.perm(omegaHat)
        omegaHat_temp = omegaHat[:, perm1][perm1, :]
        perm2 = ordering.perm(omegaHat_temp)

        if (perm2 == perm1).all():
            perm_vect = np.arange(dim)
        else:
            perm_vect = perm1

        omegaHat = omegaHat[:, perm_vect][perm_vect, :]
        data = data[:, perm_vect]

        perm_list.append(perm_vect)
        total_perm = total_perm[perm_vect]

        inverse_perm = [0] * dim
        for i, p in enumerate(total_perm):
            inverse_perm[p] = i

        omegaHatLower = np.tril(omegaHat)
        edge_count = np.count_nonzero(omegaHatLower) - dim

        for i in range(dim - 1, 1, -1):
            non_zero_ind = np.where(omegaHatLower[i, :i] != 0)[0]
            if len(non_zero_ind) > 1:
                co_parents = list(itertools.combinations(non_zero_ind, 2))
                for j in range(len(co_parents)):
                    row_index = max(co_parents[j])
                    col_index = min(co_parents[j])
                    omegaHatLower[row_index, col_index] = 1.0

        active_vars = []
        for i in range(dim):
            actives = np.where(omegaHatLower[i, :] != 0)
            active_list = list(set(actives[0]) | set([i]))
            active_list.sort(key=int)
            active_vars.append(active_list)

        n_active_vars.append(np.sum([len(x) for x in active_vars]))
        sparsity.append(n_active_vars[0] - n_active_vars[-1])

        if sparsity[-1] <= sparsity[-2]:
            sparsityIncreasing = False

        counter = counter + 1

    rec_omega = omegaHat[:, inverse_perm][inverse_perm, :]
    results_df = pd.DataFrame(results_list)

    return rec_omega, results_df