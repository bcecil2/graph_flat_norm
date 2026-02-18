import unittest

import numpy as np

from graph_flat_norm.kdtree import calculate_tree_graph,calculate_edge_vectors
from graph_flat_norm.test.test_data import unit_grid,unit_circle


class TestKDTree(unittest.TestCase):
    def test_distance_and_graph_matrix_shape(self):
        neighbors = 24
        n = 5
        domain = unit_grid(n,n)
        tree, distances, graph = calculate_tree_graph(domain,neighbors=neighbors)

        assert distances.shape == (n**2,neighbors)
        assert graph.shape == (n**2,neighbors)

    def test_graph_connections_unit_grid(self):
        """makes sure the neighbor indices in the graph output are as expected"""
        neighbors = 2
        n = 3
        domain = unit_grid(n,n)
        tree,distances,graph=calculate_tree_graph(domain,neighbors=neighbors)
        unit_grid_output = [[0,3,1],
        [1,2,0],
        [2,5,1],
        [3,4,0],
        [4,1,3],
        [5,2,4],
        [6,7,3],
        [7,4,6],
        [8,5,7]]
        assert np.allclose(unit_grid_output,graph)

    def test_graph_connections_unit_circle(self):
        """makes sure the neighbor indices in the graph output are as expected"""
        neighbors = 2
        n = 9
        domain = unit_circle(n)
        tree, distances, graph = calculate_tree_graph(domain,neighbors=neighbors)
        unit_circle_output = [[8,1],
        [2,0],
        [1,3],
        [4,2],
        [3,5],
        [6,4],
        [5,7],
        [8,6],
        [0,7]]
        assert np.allclose(unit_circle_output,graph)

    def test_edge_vectors_shape(self):
        neighbors = 24
        n = 5
        domain = unit_grid(n,n)
        tree,lengths,graph = calculate_tree_graph(domain,neighbors)
        edges,vertices = calculate_edge_vectors(domain,graph)
        assert vertices.shape == (n**2,neighbors,2)
        assert edges.shape == (n**2,neighbors,2)

    def test_edge_vectors_equality_unit_circle(self):
        neighbors = 2
        n = 9
        domain = unit_circle(n)
        tree,distances,graph=calculate_tree_graph(domain,neighbors=neighbors)
        edges,vertices = calculate_edge_vectors(domain,graph)
        expected_edges = np.array([[domain[-1]-domain[0], domain[1]-domain[0]],[domain[0]-domain[-1],domain[-2]-domain[-1]]])
        assert np.allclose(expected_edges,edges[[0,-1],:])


if __name__ == "__main__":
    unittest.main()