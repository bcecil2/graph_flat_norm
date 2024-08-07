from flat_norm import flat_norm
from test_data import unit_circle, unit_grid, grid
from test_perturb_points import perturb_points
import numpy as np
import matplotlib.pyplot as plt

def test_perimeter_circle():
    n = 100
    r = 1.0
    w = 4
    background = grid(n,n,w)
    disk = np.linalg.norm(background,axis=1)<=r
    _,perimeter,_ = flat_norm(background,disk,lamb=1,neighbors=24)
    plt.scatter(background[:,0],background[:,1])
    plt.scatter(background[disk][:,0],background[disk][:,1])
    assert np.isclose(perimeter,2*np.pi,atol=1e-1)
    
#for r1=0.5 and r2 = 0.25 one circ should disappear between 4 < lamb < 8

def test_disappearing_circles():
    n,r1,r2,w,neighbors = 125,0.5,0.25,5,24
    domain = grid(n,n,w)
    lamb = 7.9
    result = flat_norm_helper(n,r1,r2,w,neighbors,lamb,domain)
    #(True,False) = Kept circle one, did not keep circle 2
    assert circle_disppearing_helper(result, r1, r2) == (True,False)
    lamb = 8.1
    result = flat_norm_helper(n,r1,r2,w,neighbors,lamb,domain)
    assert circle_disppearing_helper(result, r1, r2) == (True,True)
    lamb = 3.9
    result = flat_norm_helper(n,r1,r2,w,neighbors,lamb,domain)
    assert circle_disppearing_helper(result, r1, r2) == (False,False)
    lamb = 4.2
    result = flat_norm_helper(n,r1,r2,w,neighbors,lamb,domain)
    assert circle_disppearing_helper(result, r1, r2) == (True,False)
    
    
def circle_disppearing_helper(result,r1,r2):
    center1 = np.array([-0.5,-0.5])
    cond1 = np.linalg.norm(result-center1,axis=1)<=r1
    center2 = np.array([0.5,0.5])
    cond2 = np.linalg.norm(result-center2,axis=1)<=r2
    return result[cond1].size != 0, result[cond2].size != 0
    
    
def flat_norm_helper(n,r1,r2,w,neighbors,lamb,domain):
    center1 = np.array([-0.5,-0.5])
    cond1 = np.linalg.norm(domain-center1,axis=1)<=r1
    center2 = np.array([0.5,0.5])
    cond2 = np.linalg.norm(domain-center2,axis=1)<=r2
    circ = cond1 + cond2
    result,perimeter,keep = flat_norm(domain,circ,lamb=lamb,neighbors=neighbors)
    return result
    
def test_disappearing_circles_perturbed():
    n = 125
    r1 = 0.5
    r2 = 0.25
    w = 5
    neighbors = 24
    lamb = 22.5
    domain = grid(n,n,w)
    domain = perturb_points(domain,0.5)
    center1 = np.array([-0.5,-0.5])
    cond1 = np.linalg.norm(domain-center1,axis=1)<=r1
    center2 = np.array([0.5,0.5])
    cond2 = np.linalg.norm(domain-center2,axis=1)<=r2
    circ = cond1 + cond2
    plt.scatter(domain[:,0],domain[:,1],label="Grid")
    plt.scatter((domain[circ])[:,0],(domain[circ])[:,1],label="original image")
    plt.legend()
    plt.figure()
    result,perimeter,keep = flat_norm(domain,circ,lamb=lamb,neighbors=neighbors)
    plt.scatter(domain[:,0],domain[:,1],label="Grid")
    plt.scatter(result[:,0],result[:,1],label="result from flat norm")
    plt.legend()
    
def test_rectangle():
    n = 125
    w = 5
    lamb=4.2
    neighbors=24
    domain = grid(n,n,w)
    x = domain[:,0]
    y = domain[:,1]
    square = (-1 <= x) & (x <= 1) & (-1 <= y) & (y <= 1)
    result,_,_ = flat_norm(domain,square,lamb=lamb,neighbors=neighbors,voronoi=True)
    plt.scatter(domain[:,0],domain[:,1])
    plt.scatter(domain[square][:,0],domain[square][:,1])
    plt.scatter(result[:,0],result[:,1])
    
if __name__ == "__main__":
    pass
    #test_perimeter_circle()
    #test_disappearing_circles()
    #test_disappearing_circles_perturbed()
    test_rectangle()