import numpy as np
from numba import jit, float64, types
import math
import os

from graph_flat_norm.sys_reducer import group_vectors

filename = "2d_lookup_table.npz"
data_path = os.path.join(os.path.dirname(__file__), filename)
file = np.load(data_path)["angle_integral_values"]

angles,values = file[:,0],file[:,1]

def get_weights(edges,lengths):
    """
    Parameters
    ----------
    edges : nparray
        N X neighbors X 2 array of vectors from each point to its neighbors
    lengths : nparray
        N X neighbors array of lengths of vectors in edges

    Returns
    -------
    weight_list : nparray
        N x neighbors list of floats corresponding to edge weights for graph

    """
    A,b,key_list, group_list = make_sys(edges,lengths)
    n = len(A)
    weight_list = []
    for i in range(n):
        weights = fast_lst_sqs(A[i],b[i])
        weight_list.append(reconstruct_system(edges[i],key_list[i],group_list[i],weights))
    weight_list = np.array(weight_list)
    return weight_list

def make_sys(edges,lengths):
    """makes an array of systems of equations [A1,A2,...] and b vectors
    [b1, b2,...] to solve Ax=b later, we group close to linearly dependent
    vectors together, and only calculate on representatives"""
    A_array = []
    b_array = []
    key_list = []
    group_list = []
    for i,system in enumerate(edges):
        angles, keys, groups = group_vectors(system)
        reduced_system = system[keys]
        n = len(reduced_system)
        sys_lengths = lengths[i][keys]
        A = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                A[i,j] = weights_numba(i,j,reduced_system,sys_lengths,values)
        A_array.append(A)
        b_array.append(4*sys_lengths)
        key_list.append(keys)
        group_list.append(groups)
    return A_array, b_array, key_list, group_list

@jit(nopython=True)
def weights_numba(i,j,u,u_lengths,values):
    length = u_lengths[i]*u_lengths[j]
    if i == j:
        return math.pi*length
    # try to jiggle into -1,1
    eps = 2.220446049250313e-16
    inner = sum([u[i][k]*u[j][k] for k in range(len(u[i]))])
    inner = inner/(length+eps)
    theta = math.acos(inner)
    idx = bs(angles,theta)
    result = length*values[idx]
    return result

@jit(types.Array(float64,1,"C")(types.Array(float64,2,"C"),types.Array(float64,1,"C")),nopython=True)
def fast_lst_sqs(A,b):
    lstsq_soln = np.linalg.lstsq(A,b)
    sing_vals = lstsq_soln[3]
    return lstsq_soln[0]

def reconstruct_system(vectors,keys,groups,weights_reduced):
    """redistributes weights amongst equivalence class for each representative"""
    weights = np.empty(len(vectors))
    weights[keys] = weights_reduced
    for key in keys:
        values = groups[key]
        n = len(values)
        if n:
            n+=1
            weights[values] = weights[key]*np.linalg.norm(vectors[key])/n
            weights[key]*=1/n
            for value in values:
                weights[value] *= 1/np.linalg.norm(vectors[value])
    return weights

@jit(nopython=True)
def bs(angles,theta):
    """binary search"""
    left,right = 0, len(angles)-1
    eps = 1e-8
    while (left <= right):
        mid = (left+ right)//2

        if abs(angles[mid] - theta) < eps:
            return mid
        elif angles[mid] < theta:
            left = mid + 1
        else:
            right = mid - 1
    return right+1