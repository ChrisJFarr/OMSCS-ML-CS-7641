import numpy as np


def get_color_graph(density, size, seed):
    np.random.seed(seed)
    edges = list()
    for i in range(0, size):
        for j in range(0, size):
            if i==j:
                continue
            if np.random.random() < density:
                edges.append((i, j))
    return edges