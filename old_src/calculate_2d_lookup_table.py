import numpy as np
#import cubepy as cp
import json
from time import time
from scipy.integrate import quad

a = 0
b = np.pi
n = 5000000

def integral(theta_uw):
    def integrand(theta,theta_uw):
        return np.abs(np.cos(theta)*np.cos(theta_uw-theta))
    result = quad(integrand, 0, 2*np.pi, args= theta_uw,epsabs=1e-16,epsrel=1e-16,limit=500)
    #print("Error estimate:", result[1])
    return result[0]

# def test_integral():
#     theta_uw = np.pi/2
#     test_integral = lambda theta:np.abs(np.cos(theta)*np.sin(theta))
#     assert np.isclose(integral(theta_uw),cp.integrate(test_integral, a,b)[0])
#     theta_uw = -np.pi/2
#     assert np.isclose(integral(theta_uw),cp.integrate(test_integral, a,b)[0])
    
#     theta_uw = np.pi
#     test_integral = lambda theta:np.abs(np.cos(theta)*np.cos(theta))
#     assert np.isclose(integral(theta_uw),cp.integrate(test_integral, a,b)[0])
    
#     theta_uw = np.pi/4
#     test_integral = lambda theta:np.abs(np.sqrt(2)/2*np.cos(theta)\
#                                         *(np.cos(theta)+np.sin(theta)))
#     assert np.isclose(integral(theta_uw),cp.integrate(test_integral, a,b)[0])
    
#     theta_uw = np.pi/2
#     test_integral = lambda theta:np.abs(1/2*(np.cos(theta)+np.sin(theta))\
#                                         *(np.cos(theta)-np.sin(theta)))
#     assert np.isclose(integral(theta_uw),cp.integrate(test_integral, a,b)[0])

#test_integral()

# def generate_list():
#     angles = np.linspace(a,b,n)
#     integrals = [integral(angle) for angle in angles]
#     result = np.column_stack((angles,integrals))
#     return result

# def save_dict(table,filename):
#     np.savetxt(filename,table,delimiter=",")

# save_dict(generate_list(),"2d_lookup_table" + str(n) + ".txt")

def generate_list():
    angles = np.linspace(a,b,n)
    integrals = [integral(angle) for angle in angles]
    result = np.column_stack((angles,integrals))
    return result

def save_dict(table,filename):
    np.savetxt(filename,table,delimiter=",")

save_dict(generate_list(),"2d_lookup_table" + "new_integrator" + "half" + str(n) + ".txt")







    
