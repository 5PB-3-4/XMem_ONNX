import math
import numpy as np
from typing import Optional
from util.torch2numpy import *


def get_similarity(mk, ms, qk, qe):
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    CK = mk.shape[1]
    mk = flatten(mk, start_dim=2)
    ms = unsqueeze(flatten(ms, start_dim=1), 2) if ms is not None else None
    qk = flatten(qk, start_dim=2)
    qe = flatten(qe, start_dim=2) if qe is not None else None

    if qe is not None:
        # See appendix for derivation
        # or you can just trust me ヽ(ー_ー )ノ
        mk = transpose(mk, 1, 2)
        a_sq = (np.power(mk, 2) @ qe)
        two_ab = 2 * (mk @ (qk * qe))
        b_sq = (qe * np.power(qk, 2)).sum(axis=1, keepdims=True)
        similarity = (-a_sq+two_ab-b_sq)
    else:
        # similar to STCN if we don't have the selection term
        a_sq = unsqueeze(np.power(mk, 2).sum(axis=1), 2)
        two_ab = 2 * (transpose(mk, 1, 2) @ qk)
        similarity = (-a_sq+two_ab)

    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)   # B*N*HW
    else:
        similarity = similarity / math.sqrt(CK)   # B*N*HW

    return similarity

def do_softmax(similarity, top_k: Optional[int]=None, inplace=False, return_usage=False):
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = topk(similarity, k=top_k, dim=1)

        x_exp = np.exp(values)
        x_exp /= np.sum(x_exp, axis=1, keepdims=True)
        if inplace:
            similarity = scatter(np.zeros_like(similarity), 1, indices, x_exp)  # B*N*HW
            affinity = similarity
        else:
            affinity = scatter(np.zeros_like(similarity), 1, indices, x_exp)  # B*N*HW
    else:
        maxes = np.max(similarity, axis=1, keepdims=True)[0]
        x_exp = np.exp(similarity - maxes)
        x_exp_sum = np.sum(x_exp, axis=1, keepdims=True)
        affinity = x_exp / x_exp_sum 
        indices = None

    if return_usage:
        return affinity, affinity.sum(axis=2)

    return affinity

def get_affinity(mk, ms, qk, qe):
    # shorthand used in training with no top-k
    similarity = get_similarity(mk, ms, qk, qe)
    affinity = do_softmax(similarity)
    return affinity

def readout(affinity, mv: np.ndarray):
    B, CV, T, H, W = mv.shape

    mo = mv.reshape(B, CV, T*H*W)
    mem = np.dot(mo, affinity)
    mo = mv.reshape(B, CV, H, W)

    return mem
