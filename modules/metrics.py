import torch
import numpy as np
from torch import nn


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size)
    )  # false rejection rates
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )  # false acceptance rates
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def get_probs(probs):
    all_labels = probs[:, -1]
    zero_index = torch.nonzero((all_labels == 0)).squeeze(-1)
    one_index = torch.nonzero(all_labels).squeeze(-1)
    return probs[zero_index, -2], probs[one_index, -2]


def cal_roc_eer(probs):
    zero_probs, one_probs = get_probs(probs)
    return compute_eer(one_probs.numpy(), zero_probs.numpy())


def acc(probs):
    zero_probs, one_probs = get_probs(probs)
    return (
        torch.concat(((zero_probs < 0.5).float(), (one_probs >= 0.5).float()))
        .mean()
        .item()
    )
