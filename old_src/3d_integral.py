import numpy as np
import matplotlib.pyplot as plt 
from sympy import *
import cubepy as cp

def spherical_to_cartesian(point):
    r = point[0]
    theta = point[1]
    phi = point[2]
    result = np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)])
    return result

def sample_sphere(n):
    u = np.random.uniform(0,1,n)
    v = np.random.uniform(0,1,n)
    #r,theta,phi
    vectors_spherical = [np.array([1.0,2*np.pi*u[i],np.arccos(2*v[i]-1)]) for i in range(n)]
    vectors_cartesian = [spherical_to_cartesian(vector) for vector in vectors_spherical]
    return np.array(vectors_cartesian)
  
def test_samples(samples):
    for entry in samples:
        assert np.allclose(1.0,np.linalg.norm(entry))

def F(x,y,z):
    return x**2+y**2

def monte_carlo_integrate(f,sample,area):
    n = len(sample)
    summands = []
    for i in range(n):
        summands.append(f(*tuple(sample[i])))
    return area/n*sum(summands)

def c3_monte_carlo_integrate(samples,u,v):
    def c3_integrand(nu):
        return np.abs(np.dot(u,nu)*np.dot(v,nu))
    n = len(samples)
    area = 4*np.pi
    summands = []
    for i in range(n):
        summands.append(c3_integrand(samples[i]))
    return area/n*sum(summands)
        

def cube_C3(vector_pair,num):
    def sphere_integrand_explicit(theta,phi):
        return np.sin(theta)*(np.abs(np.sin(theta)*np.cos(phi)*vector_pair[0][0]\
            +np.sin(theta)*np.sin(phi)*vector_pair[0][1]+np.cos(theta)*vector_pair[0][2])\
            *np.abs(np.sin(theta)*np.cos(phi)*vector_pair[1][0]\
            +np.sin(theta)*np.sin(phi)*vector_pair[1][1]+np.cos(theta)*vector_pair[1][2]))
    low = [0.0,0.0]
    high = [np.pi,2*np.pi]
    value, error = cp.integrate(sphere_integrand_explicit, low, high, itermax=num, atol=1e-6)
    return value, error


samples = sample_sphere(10**5)
u = np.array([1,1,1])
mc = c3_monte_carlo_integrate(samples, u, u)
num,_= cube_C3(np.vstack([u,u]),20)
print(mc,num,4/3*np.pi*np.linalg.norm(u)**2)
#assert(np.allclose(mc,num,atol=10**-2))

# vectors = np.random.randn(2,3)
# vectors = vectors/np.linalg.norm(vectors,axis=1)[...,np.newaxis]
# u,v = vectors[0],vectors[1]
# print(vectors)
# print(np.linalg.norm(vectors,axis=1))

# mc = c3_monte_carlo_integrate(samples, u, v)
# num,_= cube_C3(vectors,20)
# print(mc,num)
# #assert(np.allclose(mc,num,atol=10**-2))
