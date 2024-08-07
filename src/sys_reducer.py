import numpy as np
from numba import njit

#expects a list of vectors [[x1,y1],[x2,y2],...]
def group_vectors(vectors, tol = 1e-5):
    angles = find_angles(vectors)
    abs_angles = np.abs(angles)
    small_indices = abs_angles <= tol
    reflect_indices = np.abs(angles - np.pi) <= tol
    similar_indices = np.logical_or(small_indices,reflect_indices)
    groups,keys = find_groups(similar_indices)
    reduced_angles = make_reduced_angles(angles,keys)
    return reduced_angles, keys, groups

@njit
def find_angles(vectors):
    #eps = np.finfo(float).eps
    eps = 2.220446049250313e-16
    n = len(vectors)
    angles = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                angles[i,j] = np.nan
            else:
                x = vectors[i]
                y = vectors[j]
                norms = np.linalg.norm(x)*np.linalg.norm(y)+eps
                angles[i,j] = np.arccos(np.dot(x,y)/norms)
    return angles

def find_groups(similar_indices):
    seen = set()
    groups = {}
    for idx,row in enumerate(similar_indices):
        if idx not in seen:
            similar_vectors = [i for i,el in enumerate(row) if el and (i not in seen)]
            groups[idx] = similar_vectors
            seen.update(similar_vectors)
    keys = np.fromiter(groups.keys(),dtype=np.int32)
    return groups,keys

@njit
def make_reduced_angles(angles,keys):
    n = len(keys)
    result = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            result[i,j] = angles[keys[i],keys[j]]
    return result