import kdtree as kd
import voronoi as vo
import weights_system as ws
import graph_solver as gs
import numpy as np
import warnings

#number of physical cores, -1 means use all
workers = -1

def flat_norm(domain,image,lamb=1.0,neighbors = 24,voronoi=False):
    """
    Compute the flat norm for a given domain and image.

    Parameters:
    - domain (array-like): Nx2 or Nx3 array of points
    - image (array-like): Binary Nx1 mask of points in domain
    - lamb (float, optional): Curvature cutoff (default is 1.0).
    - neighbors (int, optional): Number of neighbors for edge calculations (default is 24).

    Returns:
    -  points in domain to keep, perimeter, mask of points in domain

    """
    tree,lengths,graph = kd.calculate_tree_graph(domain,neighbors,workers=workers)
    edges,vertices = kd.calculate_edge_vectors(domain,graph)
    if voronoi:
        cell_areas = vo.voronoi_areas(domain,tree,lengths,workers=workers)
    else:
        cell_areas = np.ones(len(domain))
    weights = ws.get_weights(edges, lengths)
    scaled_weights = weights*cell_areas[:,np.newaxis]
    keep,perimeter = gs.compute_flat_norm_graph_cut(graph,scaled_weights,image,cell_areas,lamb)
    if len(keep) in [0,1]:
        warnings.warn("No solution returned from min cut, lambda parameter likely too small.")
    if len(keep) >= 1:
        keep.remove("source")
    result = domain[list(keep)]
    return result, perimeter, keep