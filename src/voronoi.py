import numpy as np

def voronoi_areas(points,Tree,distances,N=1000000,workers=-1):
    """
    Parameters
    ----------
    points : nparray
        N x 2
    Tree : KDTree
    N : TYPE, Integer
        Number of sample points. The default is 1000000.
    workers : int
        Number of cores to use in KDTree. The default is -1, use all cores. 
        Change this in flat_norm.py, not here.

    Returns
    -------
    areas : nparray
        N x 1 array whose entries are the voronoi areas of each point in points

    """
    box_padding = np.mean(np.mean(distances[:,:4],axis=1))/2
    x_range, y_range, total_area = get_bounding_box(points,box_padding)
    sample_points = get_sample(x_range, y_range, N)
    nearest_neighbors = Tree.query(sample_points,workers=workers)[1]
    indices = np.zeros(len(points))
    unique,counts = np.unique(nearest_neighbors,return_counts=True)
    indices[unique] = counts
    areas = total_area/N*indices
    return areas

def get_bounding_box(points,box_padding):
    """
    find the smallest box that fully encapsulates the set of points, returns
    the x and y coordinate ranges and area of the box
    """
    x_min,x_max = np.min(points[:,0]),np.max(points[:,0])
    x_min,x_max = x_min-box_padding,x_max+box_padding
    y_min,y_max  = np.min(points[:,1]),np.max(points[:,1])
    y_min,y_max = y_min-box_padding,y_max+box_padding
    
    x_range,y_range = (x_min,x_max),(y_min,y_max)
    area = np.linalg.norm(np.diff(x_range))*np.linalg.norm(np.diff(y_range))
    return x_range, y_range, area

def get_sample(x_range,y_range,N):
    """
    return an array of points [[x1,y1], [x2,y2],...] forming a uniform grid 
    over a box
    """
    m = int(np.sqrt(N))
    sample_points_x = np.linspace(*x_range,m)
    sample_points_y = np.linspace(*y_range,m)
    return np.dstack(np.meshgrid(sample_points_x,sample_points_y)).reshape(-1,2)

