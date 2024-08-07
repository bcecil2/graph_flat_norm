from kdtree import calculate_tree_graph,calculate_edge_vectors
from test_data import unit_grid,unit_circle
import numpy as np


def test_distance_and_graph_matrix_shape():
    neighbors = 24
    n = 5
    domain = unit_grid(n,n)
    tree,distances,graph=calculate_tree_graph(domain,neighbors=neighbors)
    assert distances.shape == (n**2,neighbors)
    assert graph.shape == (n**2,neighbors+1)
    
def test_graph_connections_unit_grid():
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
    
def test_graph_connections_unit_circle():
    """makes sure the neighbor indices in the graph output are as expected"""
    neighbors = 2
    n = 9
    domain = unit_circle(n)
    tree,distances,graph=calculate_tree_graph(domain,neighbors=neighbors)
    unit_circle_output = [[0,8,1],
    [1,2,0],
    [2,1,3],
    [3,4,2],
    [4,3,5],
    [5,6,4],
    [6,5,7],
    [7,8,6],
    [8,0,7]]
    assert np.allclose(unit_circle_output,graph)

def test_edge_vectors_shape():
    neighbors = 24
    n = 5
    domain = unit_grid(n,n)
    tree,lengths,graph = calculate_tree_graph(domain,neighbors)
    edges,vertices = calculate_edge_vectors(domain,graph)
    assert vertices.shape == (n**2,neighbors,2)
    assert edges.shape == (n**2,neighbors,2)
    
def test_edge_vectors_equality_unit_circle():
    neighbors = 2
    n = 9
    domain = unit_circle(n)
    tree,distances,graph=calculate_tree_graph(domain,neighbors=neighbors)
    edges,vertices = calculate_edge_vectors(domain,graph)
    expected_edges = np.array([[domain[-1]-domain[0], domain[1]-domain[0]],[domain[0]-domain[-1],domain[-2]-domain[-1]]])
    assert np.allclose(expected_edges,edges[[0,-1],:])


if __name__ == "__main__":
    test_distance_and_graph_matrix_shape()
    test_graph_connections_unit_grid()
    test_graph_connections_unit_circle()
    test_edge_vectors_shape()
    test_edge_vectors_equality_unit_circle()