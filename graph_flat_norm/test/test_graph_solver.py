import numpy as np
import unittest

import graph_flat_norm.kdtree as kd
from graph_flat_norm.graph_solver import compile_graph
from graph_flat_norm.test.test_data import unit_circle

class TestGraphSolver(unittest.TestCase):
    def test_compile_graph_unit_circle(self):
        n = 9
        neighbors = 2
        x = unit_circle(n)
        _,_,graph = kd.calculate_tree_graph(x,neighbors)
        weights = np.arange(n*neighbors).reshape(n,neighbors)
        edges = compile_graph(graph,weights)
        assert len(edges) == n*neighbors//2
    
    
if __name__ == "__main__":
    unittest.main()
    
