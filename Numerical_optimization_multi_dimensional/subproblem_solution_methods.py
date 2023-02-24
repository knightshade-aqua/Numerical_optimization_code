import numpy as np
import math
from multi_dimentional import *
from utils import *

""" subproblem_solution_methods.py contains the functions to solve the Steihaug method of subproblem
    and the Dog-leg method """


def find_tau_dog_leg(pu, pb, delta):

    # Check whether full newton step (solution of sub-problem) lies inside the trust region
    if np.linalg.norm(pb) <= delta:
        return pb

    # Check whether the step in the steepes descent direction is greater than the trust radius.
    # If found greater then restrict and send the value of step pk
    norm_pu = np.linalg.norm(pu)
    if norm_pu >= delta:
        return delta * pu / norm_pu

    # solve ||pu +(tau-1)*(pb-pu)||**2 = trust_radius**2
    # We have to solve a quadratic equation in the variable tau   
    # Positive root of the equation is taken
    m = pu.T @ pu
    n = pb - pu
    l = n.T @ n
    q = pu.T @ n

    a = l
    b = 2 * q
    c = m - delta**2

    tau = ((-b + math.sqrt(b**2 - 4*a*c))/(2*a)) + 1

    # Return the step pk based on the value of tau
    # 0<tau<1
    if tau < 1:
        return pu*tau
    #1<tau<2
    return pu + (tau-1) * n


def subproblem_dog_leg(x_k, delta, f_k, g_k, h_k,n):

    # Obtain the objective function value, gradient and hessian at the current iterate
    f_k = f_k(x_k)
    g_k = g_k(x_k)
    h_k = h_k(x_k)

    # Find the nearest positive definite hessian if the hessian is not positive definite.
    if not is_psd(h_k):
        h_k = nearest_pd(h_k)

    # Invert the matrix
    B = np.linalg.inv(h_k)
    
    # Calculate the values of the two steps Pu and Pb
    pu = -(g_k.T @ g_k)/(g_k.T @ h_k @ g_k) * g_k
    pb = -B @ g_k

    # Obtain the value of the steps
    p_k = find_tau_dog_leg(pu,pb,delta)

    return p_k

def find_tau_cg(d_j, z_j, delta_j):

    # Solving the quadratic equation
    a = d_j.T @ d_j
    b = 2 * (z_j.T @ d_j)
    c = z_j.T @ z_j - delta_j**2 
    d = b**2 - 4*a*c
    tau = (-b + math.sqrt(d))/(2*a)

    return tau 

def subproblem_steihaug(x_k, delta_k, f_k, g_k, h_k, n):

    # Obtain the objective function value, gradient and hessian at the current iterate
    f_k = f_k(x_k)
    g_k = g_k(x_k)
    h_k = h_k(x_k)

    # Initialize the internal iterates
    z_init = np.zeros((n,1))

    # Initialize the residual
    r_init = -g_k

    # Initialize the orthogonal direction of conjugate gradient
    d_init = r_init

    # Tolerance value to stop the iteartions
    epsilon = 1e-4

    # Number of internal iterations
    N = 100

    # Check whether the chosen point is almost close to the minimum point with a tolerance of epsilon
    if (np.linalg.norm(r_init) < epsilon):
        p_k = np.zeros((n,1))
        z_init = np.zeros((n,1))
        return p_k
    
    d_j = d_init
    r_j = r_init
    z_j = z_init

    for j in range(N):

        # Check for negative curvature condition. If True calculate tau to obtain pk
        if (d_j.T @ h_k @ d_j <= 0):
            # Calculate tau to find p_k
            tau = find_tau_cg(d_j, z_j, delta_k)
            p_k = z_j + tau * d_j
            return p_k
        
        # Constant value for optimization along one dimension
        alpha_j = (r_j.T @ r_j ) / (d_j.T @ h_k @ d_j)

        # Update the internal iterates
        z_j_new = z_j + alpha_j * d_j

        # Check whether the next iterate step has overshot the trust radius
        if (np.linalg.norm(z_j_new) > delta_k):
            # find tau to find p_k
            tau = find_tau_cg(d_j, z_j, delta_k)
            p_k = z_j + tau * d_j
            return p_k
        
        # Modify the residuals
        r_j_new = r_j - alpha_j * (h_k @ d_j)

        # Check whether the gradient is almost zero with epsilon tolerance
        if ((np.linalg.norm(r_j_new))/(np.linalg.norm(g_k)) < epsilon):
            p_k = z_j_new
            return p_k
        
        # Obtain new direction d_j
        beta_j_new = (r_j_new.T @ r_j_new) / (r_j.T @ r_j)
        d_j_new = r_j_new + beta_j_new * d_j
        d_j = d_j_new
        r_j = r_j_new
        z_j = z_j_new

        ######################## Loop is repeated ######################