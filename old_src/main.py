import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import functional as f
import optimizer as opt
import graph as g  
import image_to_graph as im2g
from new_node import newnode

img_path = '25x25.png'

# def flatnorm(nodes,A,l):
#     edges = g.make_complete(nodes)
#     G = g.euclidean_graph(nodes,edges)
#     full_vector_list = []
#     for node in nodes:
#         full_vector_list += list(g.get_edge_vectors(G, node)[0])
#     norms = [np.linalg.norm(vector) for vector in full_vector_list]
#     print("minimum norm: ", min(norms))
#     opt.get_graph_weights(G)
#     g.min_cut_max_flow(G,A,l)

from time import perf_counter

def timer(fn):
    def wrapper(*args,**kwargs):
        t1 = perf_counter()
        result = fn(*args,**kwargs)
        t2 = perf_counter()
        print(f"Time to run: {t2-t1} seconds or {(t2-t1)/60} minutes")
        return result
    return wrapper

@timer
def calculate_flat_norm():
    graph,in_set = im2g.image_to_graph(img_path)
    edges = g.make_complete(graph)
    G = g.euclidean_graph(graph,edges)
    reduced = g.knn_reduced_graph(G, 24)
    opt.get_graph_weights(reduced)
    t1 = perf_counter()
    #TODO there is some bug where the source vertex gets removed twice
    #     when the parameter is here
    reduced = g.min_cut_max_flow(reduced,in_set,0.021)
    t2 = perf_counter()
    print(f"Min cut max flow took {t2-t1} seconds to run")
    xs,ys=[],[]
    try:
        reduced.remove("source")
    except None as e:
        pass
    for x,y in reduced:
        xs.append(x)
        ys.append(y)
    plt.figure()
    plt.scatter(xs, ys)

def calculate_perimeter(G,A):
    opt.get_graph_weights(G)
    s = 0
    for point in G.nodes:
        if point in A:
            point_edges = G.edges(point)
            for edge in point_edges:
                p1,p2 = edge
                if p2 not in A:
                    s+= G[p1][p2]["capacity"]
    return s

def flatnorm_circle(r,n):
    nodes = [(r*np.cos(x),r*np.sin(x)) for x in np.linspace(0,2*np.pi - .005,n)] #points on unit circle
    A = nodes[:n//2+1]
    import matplotlib.pyplot as plt
    x_val = [A[i][0] for i in range(len(A))]
    y_val = [A[i][1] for i in range(len(A))]
    plt.scatter(x_val,y_val)
    plt.show()
    e = g.make_complete(nodes)
    G = g.euclidean_graph(nodes,e)
    return calculate_perimeter(G,A)

def flatnorm_grid(n):
    nodes = [(i,j) for i in range(-n,n) for j in range(-n,n)] #grid
    A = []
    calculate_perimeter(nodes,A)

def join_origin(edge_list):
    return [((0.0,0.0),(float(u),float(v))) for u,v in edge_list]

# #making sure weights match paper weights of [0.1221,0.0476,0.0454]
# edges_black = [(1,0),(0,-1),(-1,0),(0,1)]
# edges_red = [(1,1),(1,-1),(-1,-1),(-1,1)]
# edges_blue = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]
# edges = edges_black + edges_red + edges_blue
# nodes = edges
# edges = join_origin(edges)

def split_coords(A):
    x = [entry[0] for entry in A]
    y = [entry[1] for entry in A]
    return x,y

def test_voronoi():
    x,y = np.linspace(-2,2,30), np.linspace(-2,2,30)
    #x,y = np.linspace(-1,1,3), np.array([0.0,0.0,0.0])
    nodes = [(xi,yi) for xi in x for yi in y ]
    edges = g.make_complete(nodes)
    G = g.euclidean_graph(nodes,edges)
    RG = g.knn_reduced_graph(G,24)
    E = [entry for entry in nodes if np.sqrt((entry[0])**2+(entry[1])**2) <= 1]
    plt.scatter(*split_coords(nodes))
    plt.scatter(*split_coords(E))
    plt.show()
    print(calculate_perimeter(RG,E))
    

def test_weights(nodes,edges):
    G = g.euclidean_graph(nodes,edges)
    opt.get_graph_weights(G)
    g.euclidean_plot_2d(G)
    print(nx.get_edge_attributes(G,'capacity'))

if __name__ == "__main__":
    #test_voronoi()
    calculate_flat_norm()
    plt.show()
