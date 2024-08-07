import numpy as np
import cubepy as cp
import json
from time import time

a = 0
b = 2*np.pi
n = 2*10**3

def integral(theta_uw):
    vect1 = np.array([1,0,0])
    vect2 = np.array([np.cos(theta_uw),np.sin(theta_uw),0])
    value,_ = cube_C3([vect1,vect2],25)
    return value

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

def test_integral():
    theta_uw = 10**-8
    assert np.isclose(integral(theta_uw),4/3*np.pi*np.linalg.norm(np.array([np.cos(theta_uw),np.sin(theta_uw),0])))
    
    vect1 = np.array([0,0,1])
    vect2 = np.array([np.cos(np.pi/4),np.sin(np.pi/4),np.sin(np.pi/4)])
    vect2 = vect2/np.linalg.norm(vect2)
    a = np.arccos(np.dot(vect1,vect2))
    vp = [vect1,vect2]
    print(integral(a), cube_C3(vp, 25)[0])
    assert np.isclose(integral(a), cube_C3(vp, 25)[0])
    
test_integral()

def generate_list():
    angles = np.linspace(a,b,n)
    integrals = [integral(angle) for angle in angles]
    result = np.column_stack((angles,integrals))
    return result

def save_dict(table,filename):
    np.savetxt(filename,table,delimiter=",")

#save_dict(generate_list(),"3d_lookup_table" + str(n) + ".txt")






    
