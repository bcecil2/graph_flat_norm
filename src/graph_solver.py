import networkx as nx

def compute_flat_norm_graph_cut(graph,weights,image,areas,lamb):
    """
    Parameters
    ----------
    graph : nparray
        N X neighbors list of lists indices representation of graph 
    weights : nparray
        N X neighbors each entry corresponds to entry in graph
    image : nparray
        Binary Nx1 mask of points in domain
    areas : nparray
        N X 1 voronoi cell areas corresponding to each point
    lamb : float
        curvature cutoff, change in flat_norm.py

    Returns
    -------
    keep : nparray
        mask of points to keep from domain.
    perimeter : float
        length of boundary of kept set.

    """
    edges = compile_graph(graph, weights)
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    perimeter = get_perimeter(G, image)
    add_source_sink(G,image,lamb,areas)
    keep = get_min_cut(G)
    return keep, perimeter

def compile_graph(graph,weights):
    """builds array of edges and weights, adds together weights
    on edges i->j to j->i"""
    idxs = {}
    n,m = len(graph),len(graph[0])
    assert weights.shape == (n,m)
    for i in range(n):
        for j in range(m):
            label = graph[i,j]
            if (label,i) in idxs:
                idxs[(label,i)] += weights[i,j]
            else:
                idxs[(i,label)] = weights[i,j]
                
    return [(k[0],k[1],v) for k,v in idxs.items()]

def add_source_sink(G,image,lamb,areas):
    """points in image are connected to source, points not in image are
    connected to the sink, all weighted by voronoi cell area"""
    source = "source"
    sink = "sink"
    G.add_node(source)
    G.add_node(sink)
    for i,point in enumerate(image):
        if point:
            G.add_edge(source,i,weight=lamb*areas[i])
        else:
            G.add_edge(sink,i,weight=lamb*areas[i])
            
def get_min_cut(G):
    cut_value, partition = nx.minimum_cut(G,"source","sink",capacity='weight')
    keep,_ = partition
    return keep

def get_perimeter(G,image):
    """sums weights across edges from inside image to outside image, therby
    calculating the perimeter"""
    s = 0.0
    for point in G.nodes:
        if image[point]:
            point_edges = G.edges(point)
            for edge in point_edges:
                p1,p2 = edge
                if not image[p2]:
                    s += G[p1][p2]["weight"]
    return s