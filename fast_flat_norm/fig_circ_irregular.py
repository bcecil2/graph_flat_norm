import numpy as np
import matplotlib.pyplot as plt 
import sys
from flat_norm import flat_norm
from perturb_points import perturb_points

n = 100
r1 = 0.5
r2 = 0.25
neighbors = 24
#for r1=0.5 and r2 = 0.25 one circ should disappear between 4 < lamb < 8
unperturbed_lambda= 4.2

def make_circ(n,r1,r2,neighbors,lamb,p_bound,perturb=False):
    x = np.linspace(-2,2,n)
    y = np.linspace(-2,2,n)
    points = np.dstack(np.meshgrid(x,y)).reshape(-1,2)
    if perturb:
        points = perturb_points(points,p_bound)
    plt.scatter(points[:,0],points[:,1],label="Grid")
    center1 = np.array([-0.5,-0.5])
    cond1 = np.linalg.norm(points-center1,axis=1)<=r1
    center2 = np.array([0.5,0.5])
    cond2 = np.linalg.norm(points-center2,axis=1)<=r2
    circ = cond1 + cond2
    circ_pts = (points[circ][:,0],points[circ][:,1])
    plt.scatter(*circ_pts,label="$\Omega$")
    Omega = circ
    fn_est,sigma,sigmac,perim = flat_norm(points,Omega,lamb,perim_only=False,neighbors=neighbors)
    plt.scatter(points[sigma][:,0],points[sigma][:,1],color='black',label="$\Sigma$")
    plt.legend()
    print(perim)


#sigma is the minimizing set, i.e. the thing the flat norm is really returning

make_circ(n,r1,r2,neighbors,unperturbed_lambda,0,True)

