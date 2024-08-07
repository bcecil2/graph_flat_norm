from sys_reducer import group_vectors
import numpy as np
from test_data import unit_circle

def test_group_vectors_equality_regular_grid():
    points = np.array([(-1.0,0.0),(1.0,0.0),(0.0,-1.0),(0.0,1.0)\
                  ,(-1.0,1.0),(1.0,1.0),(1.0,-1.0),(-1.0,-1.0)\
                      ,(-2.0,1.0),(-1.0,2.0),(1.0,2.0),(2.0,1.0)\
                          ,(2.0,-1.0),(1.0,-2.0),(-1.0,-2.0),(-2.0,-1.0)])
    expected_groups = {0:[1],2:[3],4:[6],5:[7],8:[12],9:[13],10:[14],11:[15]}
    reduced_angles, keys, groups = group_vectors(points)
    expected_keys = [0,2,4,5,8,9,10,11]
    assert list(keys) == expected_keys
    assert groups == expected_groups

def test_group_vectors_equality_irregular_grid():
    u = np.array([[1.0,0.0],[np.sqrt(2)/2,np.sqrt(2)/2],[0.0,1.0],
                  [-np.sqrt(2)/2,np.sqrt(2)/2],[-1.0,0.0],
                  [-np.sqrt(2)/2,-np.sqrt(2)/2],[0.0,-1.0],
                  [np.sqrt(2)/2,-np.sqrt(2)/2]])
    expected_groups = {0:[1,3,4,5,7],2:[6]}
    reduced_angles, keys, groups = group_vectors(u,tol=np.pi/4+np.finfo(float).eps)
    assert groups == expected_groups
    
def test_reduced_angles_equality():
    u = np.array([[1.0,0.0],[np.sqrt(2)/2,np.sqrt(2)/2],[0.0,1.0],
                  [-np.sqrt(2)/2,np.sqrt(2)/2],[-1.0,0.0],
                  [-np.sqrt(2)/2,-np.sqrt(2)/2],[0.0,-1.0],
                  [np.sqrt(2)/2,-np.sqrt(2)/2]])
    reduced_angles, keys, groups = group_vectors(u,tol=np.pi/4+np.finfo(float).eps)
    expected_reduced_angles = [[np.nan, np.pi/2], [np.pi/2,np.nan]]
    assert np.allclose(expected_reduced_angles,reduced_angles,equal_nan=True)
    
def test_reduced_angles_shape():
    n = 1000
    u = unit_circle(n)
    reduced_angles, keys, groups = group_vectors(u)
    assert reduced_angles.shape == (len(keys),len(keys))
    
if __name__ == "__main__":
    test_group_vectors_equality_regular_grid()
    test_group_vectors_equality_irregular_grid()
    test_reduced_angles_equality()
    test_reduced_angles_shape()
 

# import matplotlib.pyplot as plt
# u = np.array([(-1.0,0.0),(1.0,0.0),(0.0,-1.0),(0.0,1.0)\
#               ,(-1.0,1.0),(1.0,1.0),(1.0,-1.0),(-1.0,-1.0)\
#                   ,(-2.0,1.0),(-1.0,2.0),(1.0,2.0),(2.0,1.0)\
#                       ,(2.0,-1.0),(1.0,-2.0),(-1.0,-2.0),(-2.0,-1.0)])
# x = u[:,0]
# y = u[:,1]

# plt.scatter(x,y)

# for i in range(len(x)):
#     plt.annotate(i,(x[i],y[i]))