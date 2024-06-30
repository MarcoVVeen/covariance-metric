import numpy as np


def covar_metric(repres, reduced_prop, t, y0, y1, bins=40, hist_range=[[-2.5, 2.5], [-2.5, 2.5]], eps=1e-4):
    """
    Measures the confounding bias in a 2d representation repres.

    Uses a covariance metric for measuring confounding bias given a feature representation from the idea of
        deconfounding scores originally by:
    "Deconfounding Scores: Feature Representations for Causal Effect Estimation with Weak Overlap" (https://arxiv.org/abs/2104.05762)


    :param repres:  2d representation
    :param reduced_prop:    reduced propensity scores (prob. treatment given representation)
    :param t:   treatment assignments, or predicted propensity scores
    :param y0:  Y(0), or predicted outcomes under treatment
    :param y1:  Y(1), or predicted outcomes under no treatment
    :param bins:    number of bins per dimension
    :param hist_range:  range of values considered in each dimension
    :param eps:     div zero epsilon
    :return:    measured confounding bias in repres
    """
    # Calculate covariances per bin
    e_yt, _, _ = np.histogram2d(repres[:, 0], repres[:, 1], bins=bins, range=hist_range, weights=y0 * t)
    e_y, _, _ = np.histogram2d(repres[:, 0], repres[:, 1], bins=bins, range=hist_range, weights=y0)

    e_y1t, _, _ = np.histogram2d(repres[:, 0], repres[:, 1], bins=bins, range=hist_range, weights=y1 * t)
    e_y1, _, _ = np.histogram2d(repres[:, 0], repres[:, 1], bins=bins, range=hist_range, weights=y1)

    e_t, _, _ = np.histogram2d(repres[:, 0], repres[:, 1], bins=bins, range=hist_range, weights=t)
    n, _, _ = np.histogram2d(repres[:, 0], repres[:, 1], bins=bins, range=hist_range, weights=np.ones(len(t)))
    props_hist, _, _ = np.histogram2d(repres[:, 0], repres[:, 1], bins=bins, range=hist_range, weights=reduced_prop)

    mask = n > 0
    covar_bins0 = (e_yt[mask] - e_y[mask] * e_t[mask] / n[mask]) / n[mask]
    covar_bins1 = (e_y1t[mask] - e_y1[mask] * e_t[mask] / n[mask]) / n[mask]

    avg_props = props_hist[mask] / n[mask]
    covar_bins_propped = covar_bins1 / (avg_props + eps) + covar_bins0 / (1 - avg_props + eps)
    avg_covar_prop = np.mean(covar_bins_propped)

    return avg_covar_prop
