from test_data import unit_grid, unit_circle
from kdtree import calculate_tree_graph
from voronoi import voronoi_areas
import numpy as np

def voronoi_cell_areas_grid_equality():
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
    
def voronoi_cell_areas_grid_average_equality():
    n = 100
    neighbors = 1
    workers = -1
    domain = unit_grid(n,n)
    tree,_,_ = calculate_tree_graph(domain,neighbors,workers)
    cell_areas = voronoi_areas(domain,tree,workers=workers)
    assert np.mean(cell_areas) == 1/n**2
    
def vornoi_cell_areas_unit_circle_average_equality():
    n = 100**2
    neighbors = 1
    workers = -1
    domain = unit_circle(n)[:-1]
    tree,_,_ = calculate_tree_graph(domain,neighbors,workers)
    cell_areas = voronoi_areas(domain,tree,workers=workers)
    assert np.allclose(np.mean(cell_areas),4/n,atol=1e-5)

if __name__ == "__main__":
    voronoi_cell_areas_grid_equality()
    voronoi_cell_areas_grid_average_equality()
    vornoi_cell_areas_unit_circle_average_equality()