import numpy as np
import matplotlib.pyplot as plt
import unittest

from graph_flat_norm.test.test_data import grid
from graph_flat_norm.flat_norm import flat_norm, im_to_graph
from test_perturb_points import perturb_points

class TestFlatNorm(unittest.TestCase):

    def circle_disppearing_helper(self, result, r1, r2):
        center1 = np.array([-0.5, -0.5])
        cond1 = np.linalg.norm(result - center1, axis=1) <= r1
        center2 = np.array([0.5, 0.5])
        cond2 = np.linalg.norm(result - center2, axis=1) <= r2
        return result[cond1].size != 0, result[cond2].size != 0

    def flat_norm_helper(self, n, r1, r2, w, neighbors, lamb, domain):
        center1 = np.array([-0.5, -0.5])
        cond1 = np.linalg.norm(domain - center1, axis=1) <= r1
        center2 = np.array([0.5, 0.5])
        cond2 = np.linalg.norm(domain - center2, axis=1) <= r2
        circ = cond1 + cond2
        result, perimeter, keep = flat_norm(domain, circ, lamb=lamb, neighbors=neighbors)
        return result

    def test_perimeter_circle(self):
        n = 100
        r = 1.0
        w = 4
        background = grid(n,n,w)
        disk = np.linalg.norm(background,axis=1)<=r
        _, perimeter, _ = flat_norm(background,disk,lamb=1,neighbors=24,voronoi=True)
        assert np.isclose(perimeter,2*np.pi,atol=1e-1)

    #for r1=0.5 and r2 = 0.25 one circ should disappear between 4 < lamb < 8

    def test_disappearing_circles(self):
        n,r1,r2,w,neighbors = 125,0.5,0.25,5,24
        domain = grid(n,n,w)
        lamb = 7.9
        result = self.flat_norm_helper(n,r1,r2,w,neighbors,lamb,domain)
        #(True,False) = Kept circle one, did not keep circle 2
        assert self.circle_disppearing_helper(result, r1, r2) == (True,False)
        lamb = 8.1
        result = self.flat_norm_helper(n,r1,r2,w,neighbors,lamb,domain)
        assert self.circle_disppearing_helper(result, r1, r2) == (True,True)
        lamb = 3.9
        result = self.flat_norm_helper(n,r1,r2,w,neighbors,lamb,domain)
        assert self.circle_disppearing_helper(result, r1, r2) == (False,False)
        lamb = 4.2
        result = self.flat_norm_helper(n,r1,r2,w,neighbors,lamb,domain)
        assert self.circle_disppearing_helper(result, r1, r2) == (True,False)




    def test_disappearing_circles_perturbed(self):
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

    def test_rectangle(self):
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
        plt.show()

    def test_im_to_graph(self):
        domain,image = im_to_graph("50x50.png")
        lamb = 15
        neighbors = 4
        result, _, _ = flat_norm(domain, image, lamb=lamb, neighbors=neighbors, voronoi=True)
        # magic number just checking that some of the spikes get cut off
        assert result.shape[0] == 713

if __name__ == "__main__":
    unittest.main()