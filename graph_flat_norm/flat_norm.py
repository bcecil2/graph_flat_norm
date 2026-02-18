import numpy as np
import cv2
import warnings

import graph_flat_norm.kdtree as kd
import graph_flat_norm.voronoi as vo
import graph_flat_norm.weights_system as ws
import graph_flat_norm.graph_solver as gs


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
        cell_areas = vo.voronoi_areas(domain,tree,distances=lengths,workers=workers)
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

def im_to_graph(im_path):
    """
    Turn an image into a characteristic function.

    Parameters:
    - im_path: path to image

    Returns:
    -  background grid for whole image, mask of points in domain
    """
    def img_to_cartesian(x, y, h):
        x, y = y, h - x
        return x, y
    img = cv2.imread(im_path, 0) + 0.
    img /= 255
    img = np.where(img >= 0.5, 0, 1)
    image_width, image_height = np.shape(img)[1],np.shape(img)[0]
    rows,cols = np.indices(np.shape(img))
    rows = rows.flatten()
    cols = cols.flatten()
    grid = img_to_cartesian(rows, cols, image_height)
    max_x, max_y = np.max(grid[0]), np.max(grid[1])
    grid_x,grid_y = grid[0] / max_x, grid[1] / max_y
    domain = np.stack((grid_x, grid_y), axis=-1)

    image = img.flatten() != 0
    return domain, image