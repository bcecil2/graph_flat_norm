import numpy as np

def unit_grid(n,m):
    points_x = np.linspace(-0.5, 0.5, n)
    points_y = np.linspace(-0.5, 0.5, m)
    points = np.dstack(np.meshgrid(points_x,points_y)).reshape((-1,2))
    return points

def unit_circle(n):
    t = np.linspace(0,2*np.pi,n)
    points_x = np.cos(t)[:,np.newaxis]
    points_y = np.sin(t)[:,np.newaxis]
    points = np.hstack([points_x,points_y])
    return np.array(points)

def grid(n,m,w):
    points_x = np.linspace(-0.5*w, 0.5*w, n)
    points_y = np.linspace(-0.5*w, 0.5*w, m)
    print(points_x[1]-points_x[0])
    points = np.dstack(np.meshgrid(points_x,points_y)).reshape((-1,2))
    return points