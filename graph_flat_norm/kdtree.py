from scipy.spatial import KDTree
import numpy as np


def calculate_tree_graph(points,neighbors=24,workers=-1):
    """
    Parameters
    ----------
    points : nparray
        N x 2 entire domain
    neighbors : int
        Number of neighbors for the graph and tree. The default is 24.
    workers : int
        Number of cores to use in KDTree. The default is -1, use all cores. 
        Change this in flat_norm.py, not here.

    Using the KDTree compute the distances and graph for the points. A point
    cannot be a neighbor of itself.
    
    Indices for the graph are the same as points.
    [[neighbors of point[0]], [neighbors of point[1]],...]

    Returns
    -------
    tree : KDTree
        Used for Voronoi calculations later.
    distances : nparray
        N X neighbors distance matrix
    graph : nparray
        N X neighbors list of lists indices representation of graph 
    """
    tree = KDTree(points)
    distances,graph = tree.query(points,neighbors+1,workers=workers)
    distances = distances[:,1:]
    graph = graph[:,1:]
    return tree,distances,graph

def calculate_edge_vectors(points,graph):
    """
    For each vertex in the graph, compute edge vectors to neighbors.
    """
    vertices = points[graph.astype(np.int32)]
    edges = vertices - points[:,np.newaxis] #subtract to get vectors
    return edges,vertices