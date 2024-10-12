import numpy as np


# Soft aggregation from STM
def aggregate(prob, dim, return_logits=False):
    new_prob = np.concatenate([
        np.prod(1-prob, axis=dim, keepdims=True),
        prob
    ], dim).clip(1e-7, 1-1e-7)
    logits = np.log((new_prob /(1-new_prob)))
    prob = np.exp(logits)/np.sum(np.exp(logits), axis=dim)

    if return_logits:
        return logits, prob
    else:
        return prob