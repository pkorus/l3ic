from collections import Counter
import numpy as np


def short_string(d, n=40):
    n2 = n // 2 - 2
    if len(d) > n:
        return ''.join((d[:n2].decode('utf8')) + ' .. ' + (d[-n2:].decode('utf8')))
    else:
        return str(d)


def entropy(p):
    if type(p) is not np.ndarray:
        p = np.array(list(p))
    return - float(np.sum(p * np.log2(p)))


def symbol_probabilities(data):
    n = len(data)
    return {k: v/n for k, v in dict(Counter(data)).items()}
