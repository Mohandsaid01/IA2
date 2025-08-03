import numpy as np

def calculate_distance(vec1, vec2, metric):
    if metric == 'euclidean':
        return np.linalg.norm(vec1 - vec2)
    elif metric == 'manhattan':
        return np.sum(np.abs(vec1 - vec2))
    elif metric == 'chebyshev':
        return np.max(np.abs(vec1 - vec2))
    elif metric == 'canberra':
        return np.sum(np.abs(vec1 - vec2) / (np.abs(vec1) + np.abs(vec2) + 1e-10))
    else:
        raise ValueError("Unknown distance metric")
