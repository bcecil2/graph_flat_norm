import pstats

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
import warnings
from scipy.sparse import csr_array
from functools import wraps
from time import perf_counter
import cProfile
from numba import njit, jit, int32, float64, types
import math

#number of physical cores
workers = 4

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        print ('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap

points_x = np.linspace(-2,2,1000)
points_y = np.linspace(-2,2,1000)

# points = np.dstack(np.meshgrid(points_x,points_y)).reshape((-1,2))

# points_disk = np.linalg.norm(points,axis=1)<=1

# plt.scatter(points[:,0],points[:,1])
# plt.scatter(points[points_disk][:,0],points[points_disk][:,1])

# t1 = perf_counter()

# =============================================================================
# above this is temporary testing stuff only
# =============================================================================

#filename = '2d_lookup_table100000.txt'

#filename = '2d_lookup_table5000.txt'

#filename = "2d_lookup_tablenew_integrator50000.txt"

#filename = "2d_lookup_tablenew_integratorhalf50000.txt"

filename = "2d_lookup_tablenew_integratorhalf5000000.txt"

file = np.loadtxt(filename,delimiter=',')

angles,values = file[:,0],file[:,1]

rng = np.random.default_rng(2025)

@jit(nopython=True)
def bs(angles,theta):
    left,right = 0, len(angles)-1
    eps = 1e-8
    while (left <= right):
        mid = (left+ right)//2

        if abs(angles[mid] - theta) < eps:
            return mid
        elif angles[mid] < theta:
            left = mid + 1
        else:
            right = mid - 1
    return right+1

@timing
def get_perimeter(E,G):
    #measure function
    s = 0.0
    for point in G.nodes:
        if E[point]:
            point_edges = G.edges(point)
            for edge in point_edges:
                p1,p2 = edge
                if not E[p2]:
                    s += G[p1][p2]["weight"]
    return s

@jit(nopython=True)
def weights_numba(i,j,u,u_lengths,values):
    #weight_function
    length = u_lengths[i]*u_lengths[j]
    if i == j:
        return math.pi*length
    # try to jiggle into -1,1
    eps = 2.220446049250313e-16
    inner = sum([u[i][k]*u[j][k] for k in range(len(u[i]))])
    inner = inner/(length+eps)
    theta = math.acos(inner)
    idx = bs(angles,theta)
    result = length*values[idx]
    return result

@jit(types.Array(float64,1,"C")(types.Array(float64,2,"C"),types.Array(float64,1,"C")),nopython=True)
def fast_lst_sqs(A,b):
    lstsq_soln = np.linalg.lstsq(A,b)
    sing_vals = lstsq_soln[3]
    #print("cond number:", np.max(sing_vals)/np.min(sing_vals))
    return lstsq_soln[0]

eps = np.finfo(float).eps

@njit
def find_angles(vectors):
    n = len(vectors)
    angles = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                angles[i,j] = np.nan
            else:
                x = vectors[i]
                y = vectors[j]
                norms = np.linalg.norm(x)*np.linalg.norm(y)+eps
                angles[i,j] = np.arccos(np.dot(x,y)/norms)
    return angles

@njit
def make_reduced_angles(angles,keys):
    n = len(keys)
    result = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            result[i,j] = angles[keys[i],keys[j]]
    return result

def find_groups(similar_indices):
    seen = set()
    groups = {}
    for idx,row in enumerate(similar_indices):
        if idx not in seen:
            similar_vectors = [i for i,el in enumerate(row) if el]
            groups[idx] = similar_vectors
            seen.update(similar_vectors)
    keys = np.fromiter(groups.keys(),dtype=np.int32)
    return groups,keys

def group_vectors(vectors, tol = 1e-5):
    angles = find_angles(vectors)
    abs_angles = np.abs(angles)
    small_indices = abs_angles <= tol
    reflect_indices = np.abs(angles - np.pi) <= tol
    similar_indices = np.logical_or(small_indices,reflect_indices)
    groups,keys = find_groups(similar_indices)
    reduced_angles = make_reduced_angles(angles,keys)
    return reduced_angles, keys, groups

def reconstruct_system(vectors,keys,groups,weights_reduced):
    weights = np.empty(len(vectors))
    weights[keys] = weights_reduced
    #print(keys)
    #print(weights_reduced)
    for key in keys:
        old_weight = weights[key]
        values = groups[key]
        #print(vectors[values])
        #print(vectors[key])
        n = len(values)
        if n:
            n+=1
            weights[values] = weights[key]*np.linalg.norm(vectors[key])/n
            weights[key]*=1/n
            # s = weights[key]*np.linalg.norm(vectors[key])
            for value in values:
                weights[value] *= 1/np.linalg.norm(vectors[value])
                # s+= np.linalg.norm(vectors[value])*weights[value]
            # print("WEIGHT VS SUM")
            # print(old_weight*np.linalg.norm(vectors[key]))
            # print(s)
    return weights

def make_sys(edges,lengths):
    A_array = []
    b_array = []
    key_list = []
    group_list = []
    for i,system in enumerate(edges):
        angles, keys, groups = group_vectors(system)
        reduced_system = system[keys]
        # print(system)
        # print(groups)
        # print(keys)
        # print(reduced_system)
        n = len(reduced_system)
        sys_lengths = lengths[i][keys]
        # print(sys_lengths)
        A = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                A[i,j] = weights_numba(i,j,reduced_system,sys_lengths,values)
        A_array.append(A)
        b_array.append(4*sys_lengths)
        key_list.append(keys)
        group_list.append(groups)
    return A_array, b_array, key_list, group_list

@timing
def get_weights(edges,lengths):
    A,b,key_list, group_list = make_sys(edges,lengths)
    n = len(A)
    weight_list = []
    for i in range(n):
        weights = fast_lst_sqs(A[i],b[i])
        weight_list.append(reconstruct_system(edges[i],key_list[i],group_list[i],weights))
    return weight_list

def get_sample(x_range,y_range,N):
    m = int(np.sqrt(N))
    sample_points_x = np.linspace(*x_range,m)
    sample_points_y = np.linspace(*y_range,m)
    #sample_points_x = rng.uniform(*x_range,N)
    #sample_points_y = rng.uniform(*y_range,N)
    #return np.column_stack((sample_points_x,sample_points_y))
    return np.dstack(np.meshgrid(sample_points_x,sample_points_y)).reshape(-1,2)

def get_bounding_box(points):
    x_min,x_max = np.min(points[:,0]),np.max(points[:,0])
    y_min,y_max  = np.min(points[:,1]),np.max(points[:,1])
    x_range,y_range = (x_min,x_max),(y_min,y_max)
    area = np.linalg.norm(np.diff(x_range))*np.linalg.norm(np.diff(y_range))
    return x_range, y_range, area

@timing
def voronoi_areas(points,Tree,N=1000000):
    x_range, y_range, total_area = get_bounding_box(points)
    sample_points = get_sample(x_range, y_range, N)
    nearest_neighbors = Tree.query(sample_points,workers=workers)[1]
    print(nearest_neighbors)
    indices = np.zeros(len(points))
    unique,counts = np.unique(nearest_neighbors,return_counts=True)
    indices[unique] = counts
    areas = total_area/N*indices
    return areas

@timing
def calculate_tree_graph(points,neighbors=24):
    Tree = KDTree(points)
    graph = np.array(Tree.query(points,neighbors+1,workers=workers))
    neighbor_indices = graph[1,:,1:]
    n = len(points)
    i,j = np.indices((n,neighbors))
    weight_indices = np.dstack((i,neighbor_indices)).reshape(-1,2).astype(np.int32)
    return Tree,graph,weight_indices

def calculate_edge_vectors(points,graph):
    lengths = graph[0,:,1:] #trim off the first column (all 0s)
    vertices = points[graph[1,:,1:].astype(np.int32)] #ditto as above
    edges = vertices - points[:,np.newaxis] #subtract to get vectors
    #print("neighbor edges:", edges[len(edges)//2+2])
    #print("vertex:", vertices[len(edges)//2+2])
    return edges,lengths,vertices

def add_source_sink(G,E,lamb,areas):
    source = "source"
    sink = "sink"
    G.add_node(source)
    G.add_node(sink)
    for i,point in enumerate(E):
        if point:
            G.add_edge(source,i,weight=lamb*areas[i])
        else:
            G.add_edge(sink,i,weight=lamb*areas[i])

@timing
def get_min_cut(G):
    return nx.minimum_cut(G,"source","sink",capacity='weight')

def flat_norm(points,E,lamb=1.0,perim_only=False,neighbors = 24):
    #main function
    n = len(points)
    if neighbors >= len(points):
        raise Exception("Need more points than neighbors")
    weightst0 = perf_counter()
    Tree,graph,weight_indices = calculate_tree_graph(points,neighbors)
    edges,lengths,vertices = calculate_edge_vectors(points,graph)
    areas = voronoi_areas(points,Tree)
    #sorted_areas = np.sort(areas)
    #vals,idx,occ = np.unique(sorted_areas,return_index=True,return_counts=True)
    #vals = np.round(vals,4)
    
    weights = get_weights(edges, lengths)
    areas = 0.0016*np.ones(np.shape(areas))
    #scaled_weights = np.multiply(weights,areas[:,np.newaxis]).flatten()
    scaled_weights = (weights*areas[:,np.newaxis]).flatten()

    weightst1 = perf_counter()
    perimt0 = perf_counter()
    
    row = weight_indices[:, 0]
    col = weight_indices[:, 1] 
    graph_weights = scaled_weights
    sparse = csr_array((graph_weights, (row, col)), shape=(n, n))
    #sparse = sparse + sparse.T
    G = nx.from_scipy_sparse_array(sparse)
    
    perim = get_perimeter(E,G)
    perimt1 = perf_counter()

    if perim_only:
        return None,None,None,perim

    mft0 = perf_counter()
    add_source_sink(G,E,lamb,areas)
    cut_value, partition = get_min_cut(G)
    keep,_ = partition
    mft1 = perf_counter()

    times = [mft1-mft0,weightst1 - weightst0,perimt1 - perimt0]
    total = sum(times)
    # print(f"Time to finish weights\n raw: {weightst1 - weightst0}\n %: {(weightst1 - weightst0) / total}")
    # print()
    # print(f"Time to finish perim\n raw: {perimt1 - perimt0}\n %: {(perimt1 - perimt0) / total}")
    # print()
    #print(f"Time to finish min cut\n raw: {mft1 - mft0}\n %: {(mft1 - mft0) / total}")
    if len(keep) in [0,1]:
        warnings.warn("No solution returned from min cut, lambda parameter likely too small.")
    if len(keep) >= 1:
        keep.remove("source")

    result = points[list(keep)]
    #plt.scatter(result[:,0],result[:,1])
    sigma = np.zeros(n).astype(bool)
    sigma[list(keep)] = 1
    sigma = sigma.astype(bool)
    sigmac = ~sigma
    return result, sigma, sigmac, perim

# =============================================================================
# below this is temporary testing stuff only
# =============================================================================

#flat_norm(points,points_disk,lamb=1e-2,neighbors=8)

# print(perf_counter() - t1)

# =============================================================================
# validating Kevin's weights
# =============================================================================

if __name__ == "__main__":
    # u = np.array([(0.0,0.0),(-1.0,0.0),(1.0,0.0),(0.0,-1.0),(0.0,1.0)\
    #               ,(-1.0,1.0),(1.0,1.0),(1.0,-1.0),(-1.0,-1.0)\
    #                   ,(-2.0,1.0),(-1.0,2.0),(1.0,2.0),(2.0,1.0)\
    #                       ,(2.0,-1.0),(1.0,-2.0),(-1.0,-2.0),(-2.0,-1.0)])
    # points = np.array([(0.0,0.0),(-1.0,0.0),(0.0, -1.0)\
    #               ,(-1.0,1.0),(1.0,1.0)\
    #                   ,(-2.0,1.0),(-1.0,2.0),(1.0,2.0),(2.0,1.0)\
    #                       ])
    points_x = np.arange(-5, 5, 1, dtype=np.float64)
    points_y = np.arange(-5, 5, 1, dtype=np.float64)
    points = np.dstack(np.meshgrid(points_x,points_y)).reshape(-1,2)
    #plt.scatter(points[:,0],points[:,1])
    #u_lengths = np.linalg.norm(u,axis=1)
    flat_norm(points,np.ones(len(points)),lamb=1.0,neighbors=8)
    #result = get_weights(u,u_lengths)
    #print("Ours: ", [result[0],result[5],result[9]])
    print("KRV: ", [0.1221, 0.0476, 0.0454])

    # tick = perf_counter()
    # flat_norm(points, points_disk, lamb=1, neighbors=8)
    # tock = perf_counter()
    # print(tock-tick)
    import pstats
    from pstats import SortKey

# =============================================================================
#     points_x = np.linspace(-2, 2, 100)
#     points_y = np.linspace(-2, 2, 100)
#     points = np.dstack(np.meshgrid(points_x,points_y)).reshape((-1,2))
# 
#     points_disk = np.linalg.norm(points,axis=1)<=1
#     flat_norm(points, points_disk, lamb=1.0, neighbors=24)
# =============================================================================
    #cProfile.run('flat_norm(points, points_disk, lamb=.001, neighbors=8)',"flatnorm")
    #p = pstats.Stats("flatnorm")
    #p.strip_dirs().sort_stats(SortKey.TIME).print_stats(50)