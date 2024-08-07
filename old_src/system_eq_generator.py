from sympy import *
import numpy as np
from time import time
import functional as fu

# c1 = symbols("1c:4")
# c2 = symbols("2c:4")
# c3 = symbols("3c:6")
# w = symbols("w:4")

# dictionary = {"12":0, "13":1,"14":2,"21":0,"23":3,"24":4,"31":1,"32":3,"34":5,"41":2,"42":4,"43":5}

# f = lambda x: dictionary[x]

def numerical_vs_exact(u,eqns,w,d):
    """get and solve the lin system"""

    t0 = time()
    A,b = fu.construct_weights_system(u)

    num_soln = np.linalg.solve(A,b)
    t1 = time()
    #print("Solved Matrix in:", t1-t0)
    
    """get weights and plug into exact soln"""

    c1_list = fu.c1_2d(u)

    c2_list = fu.c2_2d(u)

    c3_matrix = [[float(fu.c3_2d_integral(u, k, m, {})) for k in range(len(u))] for m in range(len(u))]

    c1_dict = {"1c"+str(i): c1_list[i] for i in range(len(c1_list))}

    c2_dict = {"2c"+str(i): c2_list[i] for i in range(len(c2_list))}

    f = lambda x: d[x]

    c3_dict = {"3c"+str(f(str(k)+str(m))): c3_matrix[k][m] for k in range(len(u)) for m in range(len(u)) if k != m }

    const_dict = c1_dict | c2_dict | c3_dict

    assert len(const_dict) == sum(len(u) for u in [c1_dict,c2_dict,c3_dict])
    t0 = time()
    solns = list(linsolve(eqns,*w))[0]
    vardict = {w[i]:solns[i] for i in range(len(solns))}
    t1 = time()
    #print("Solved System in:", t1-t0)
    exact_soln = np.array([float(vardict[w[i]].subs(const_dict)) for i in range(len(solns))])

    assert np.allclose(num_soln,exact_soln)
    return






