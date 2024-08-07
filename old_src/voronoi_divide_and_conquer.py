import numpy as np
import matplotlib.pyplot as plt 

from time import perf_counter

#sample_points = [np.array([1.0,2.0]),np.array([1.0,0.0]),np.array([0.0,0.0]),np.array([0.0,1.0])]


#sample_points = [(float(x),float(y)) for x in range(-1,2) for y in range(-1,2)]

x,y = np.linspace(-2,2,30), np.linspace(-2,2,30)
sample_points = [(xi,yi) for xi in x for yi in y ]

# theta = np.pi/4
# A = np.array([[np.cos(theta), -np.sin(theta)],
#               [np.sin(theta),np.cos(theta)]])
# sample_points = np.array(sample_points)
# sample_points = [A @ point for point in sample_points]
# sample_points = [tuple(entry) for entry in sample_points]

def split_coords(A):
    x = [entry[0] for entry in A]
    y = [entry[1] for entry in A]
    return x,y

def get_subdivision(points):
    """
    Parameters
    ----------
    p1-p4 : nparrays, 
        in clockwise order from top right corner

    Returns
    -------
    None.
    """
    p1,p2,p3,p4 = points
    mid_height = p2[1]+0.5*(p1[1]-p2[1])
    mid_width = p4[0]+0.5*(p1[0]-p4[0])
    right_side = p1[0]
    top_side = p1[1]
    bot_side = p2[1]
    left_side = p4[0]
    box_1 = [p1,(right_side,mid_height),(mid_width,mid_height),(mid_width,top_side)]
    box_2 = [(right_side,mid_height),p2,(mid_width,bot_side),(mid_width,mid_height)]
    box_3 = [(mid_width,mid_height),(mid_width,bot_side),p3,(left_side,mid_height)]
    box_4 = [(mid_width,top_side),(mid_width,mid_height),(left_side,mid_height),p4]
    return [box_1,box_2,box_3,box_4]
    

def get_bounding_box(points):
    """finds the smallest box containing the points and its area"""
    x_coords, y_coords = zip(*points)
    x_range = (np.min(x_coords),np.max(x_coords))
    y_range = (np.min(y_coords),np.max(y_coords))
    area = np.linalg.norm(np.diff(x_range))*np.linalg.norm(np.diff(y_range))
    return [(x_range[1],y_range[1]),(x_range[1],y_range[0]),(x_range[0],y_range[0]),(x_range[0],y_range[1])]

def calculate_voronoi_area(points):
    bounding_box = get_bounding_box(points)
    point_areas = {tuple(point):0 for point in points}
    """
    Parameters
    ----------
    points : list
        list of nparrays representing a set of pts in R2

    Returns
    -------
    dictionary of points w/ their associated areas 

    """
    t1 = perf_counter()
    recurse(bounding_box,points,point_areas)
    print(perf_counter()-t1)
    return point_areas

def calc_dist(x,y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

def nearest_point(point,points):
    nearest = np.argmin([calc_dist(point,entry) for entry in points])
    return points[nearest]

def box_area(box):
    p1 = box[0]
    p2 = box[1]
    p3 = box[2]
    return (p1[1]-p2[1])*(p2[0]-p3[0])

def recurse(box,points,point_areas):
    nearest = nearest_point(box[0],points)
    descend = False
    for point in box[1:]:
        if nearest_point(point,points) != nearest:
            descend = True
            break
    if box_area(box) <= 10**(-3):
        descend = False
    elif descend:
        boxes = get_subdivision(box)
        for subbox in boxes:
            recurse(subbox,points,point_areas)
    point_areas[nearest] += (not descend)*box_area(box)
    return
