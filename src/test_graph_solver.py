import kdtree as kd
import numpy as np
from graph_solver import compile_graph
from test_data import unit_circle, unit_grid

def test_compile_graph_unit_circle():
    n = 9
    neighbors = 2
    x = unit_circle(n)
    _,_,graph = kd.calculate_tree_graph(x,neighbors)
    weights = np.arange(n*neighbors).reshape(n,neighbors)
    edges = compile_graph(graph,weights)
    assert len(edges) == n*neighbors//2
    
    
if __name__ == "__main__":
    test_compile_graph_unit_circle()
    
