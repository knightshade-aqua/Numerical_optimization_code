import numpy as np
import objective_function_grad
import math
from hessian_func import hessian

""" subproblem_solution.py contains the method to solve steihaug method """

def find_tau(d_j, z_j, delta_j):
    # Solving the quadratic equation
    a = d_j.T @ d_j
    b = 2 * (z_j.T @ d_j)
    c = z_j.T @ z_j - delta_j**2 
    tau = (-b + math.sqrt(b**2 - 4*a*c))/(2*a)
    return tau 

def subproblem_steihaug(x_k, delta_k, objective):
    if objective == "rosenbrock":
        g_k = objective_function_grad.rosenbrock(x_k)
    elif objective == "booth":
        g_k = objective_function_grad.booth(x_k)
    else:
        g_k = objective_function_grad.ackley(x_k)

    h_k = hessian(x_k, objective)

    # Initialize the internal iterates
    z_init = np.zeros(2)

    # Initialize the residual
    r_init = -g_k

    # Initialize the orthogonal direction of conjugate gradient
    d_init = r_init
    epsilon = 1e-4
    N = 10

    # Check whether the chosen point is almost close to the minimum point with a tolerance of epsilon
    if (np.linalg.norm(r_init) < epsilon):
        p_k = np.zeros(2)
        z_init = np.zeros(2)
        return p_k
    
    d_j = d_init
    r_j = r_init
    z_j = z_init

    for j in range(N):

        # Check for negative curvature condition
        if (d_j.T @ h_k @ d_j <= 0):
            # Calculate tau to find p_k
            tau = find_tau(d_j, z_j, delta_k)
            p_k = z_j + tau * d_j
            
            return p_k
        
        # Calculate next iterate step
        alpha_j = (r_j.T @ r_j ) / (d_j.T @ h_k @ d_j)
        z_j_new = z_j + alpha_j * d_j

        # Check whether the next iterate step has overshot the trust radius
        if (np.linalg.norm(z_j_new) > delta_k):
            # find tau to find p_k
            tau = find_tau(d_j, z_j, delta_k)
            p_k = z_j + tau * d_j
            return p_k
        
        # Modify the gradient
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

        #print("############# Loop repeated ###############")