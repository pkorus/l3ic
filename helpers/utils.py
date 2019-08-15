import numpy as np


def bin_egdes(code_book):
    max_float = np.abs(code_book).max() * 2
    code_book_edges = np.convolve(code_book, [0.5, 0.5], mode='valid')
    code_book_edges = np.concatenate((-np.array([max_float]), code_book_edges, np.array([max_float])), axis=0)
    return code_book_edges


def qhist(values, code_book, density=False):
    code_book_edges = bin_egdes(code_book)
    return np.histogram(values.reshape((-1, )), bins=code_book_edges, density=density)[0]


def entropy(batch_z, code_book):
    counts = qhist(batch_z, code_book)
    counts = counts.clip(min=1)
    probs = counts / counts.sum()
    return - np.sum(probs * np.log2(probs))
