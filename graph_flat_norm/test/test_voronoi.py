import unittest

import numpy as np

from graph_flat_norm.test.test_data import unit_grid, unit_circle
from graph_flat_norm.kdtree import calculate_tree_graph
from graph_flat_norm.voronoi import voronoi_areas


class TestVoronoi(unittest.TestCase):
    def test_voronoi_cell_areas_grid_equality(self):
        n = 3
        neighbors = 3
        workers = -1
        domain = unit_grid(n,n)
        tree,_,_ = calculate_tree_graph(domain,neighbors,workers)
        cell_areas = voronoi_areas(domain,tree,workers=workers)

        assert cell_areas[0] == cell_areas[-1]
        assert cell_areas[1] == cell_areas[-2]
        assert cell_areas[2] == cell_areas[-3]

        for i in range(n):
            assert cell_areas[i] == cell_areas[-1-i]
        assert cell_areas[n**2//2] == 0.25

    def test_voronoi_cell_areas_grid_average_equality(self):
        n = 100
        neighbors = 1
        workers = -1
        domain = unit_grid(n,n)
        tree,_,_ = calculate_tree_graph(domain,neighbors,workers)
        cell_areas = voronoi_areas(domain,tree,workers=workers)
        assert np.mean(cell_areas) == 1/n**2

    def test_vornoi_cell_areas_unit_circle_average_equality(self):
        n = 100**2
        neighbors = 1
        workers = -1
        domain = unit_circle(n)[:-1]
        tree,_,_ = calculate_tree_graph(domain,neighbors,workers)
        cell_areas = voronoi_areas(domain,tree,workers=workers)
        assert np.allclose(np.mean(cell_areas),4/n,atol=1e-5)

if __name__ == "__main__":
    unittest.main()