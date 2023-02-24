import numpy as np
import objective_function_grad
import math
from hessian_func import hessian
from utils import *

def find_tau(pu, pb, delta):
    
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

    # 0<tau<1
    if tau < 1:
        return pu*tau
    #1<tau<2
    return pu + (tau-1) * n

def subproblem_dog_leg(x_k, delta, objective):

    if objective == "rosenbrock":
        g_k = objective_function_grad.rosenbrock(x_k)
    elif objective == "booth":
        g_k = objective_function_grad.booth(x_k)
    else:
        g_k = objective_function_grad.ackley(x_k)

    h_k = hessian(x_k, objective)

     # Find the nearest positive definite hessian if the hessian is not positive definite.
    if not is_psd(h_k):
        h_k = nearest_pd(h_k)
        
    # Calculate the values of the two steps Pu and Pb
    pu = -(g_k.T @ g_k)/(g_k.T @ h_k @ g_k) * g_k
    B = np.linalg.inv(h_k)
    pb = -B @ g_k

    p_k = find_tau(pu,pb,delta)
    return p_k


