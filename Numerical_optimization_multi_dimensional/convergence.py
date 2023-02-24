import numpy as np
from statistics import mean

""" convergence.py does the calculations to obatain the order Q values and constant C values 
    to as present in the convergence rate formulas in the report """

# Calculate C value for quadratic convergence
# The calculation has been performed according to the convergence rate formula for Q quadratic convergence
def iter_quad(a, minima):
    b = np.repeat(minima, repeats=a.shape[1], axis=1)
    a = a - b
    a = np.linalg.norm(a, axis=0)
    res = [(y/x) for x,y in zip(a[:-1], a[1:])]

    # Mean is taking to obtain a single C value over the iteartions
    res = mean(res)
    return res

# Calculate C value for quadratic convergence
def calculate_convergence_quad(trajectory_cg, trajectory_dl, trajectory_cp, n):

    # Initialize minimum values for the different methods
    minima_cg = np.ones((n,1))
    minima_dl = np.ones((n,1))
    minima_cp = np.ones((n,1))

    # Obtain C values for linear convergence rates
    out_cg = iter_quad(trajectory_cg.T, minima_cg)
    out_dl = iter_quad(trajectory_dl.T, minima_dl)
    out_cp = iter_quad(trajectory_cp.T, minima_cp)

    return out_cg, out_dl, out_cp

# Calculate C value for linear convergence
# The calculation has been performed according to the convergence rate formula for Q linear and superlinear convergence
def iter_lin(a, minima):
    b = np.repeat(minima, repeats=a.shape[1], axis=1)
    a = a - b
    a = np.linalg.norm(a, axis=0)
    res = [(y/x) for x,y in zip(a[:-1], a[1:])]

    # Mean is taking to obtain a single C value over the iteartions
    res = mean(res)
    return res

# Calculate C value for linear convergence
def calculate_convergence_lin(trajectory_cg, trajectory_dl, trajectory_cp, n):

    # Initialize minimum values for the different methods
    minima_cg = np.ones((n,1))
    minima_dl = np.ones((n,1))
    minima_cp = np.ones((n,1))

    # Obtain C values for linear convergence rates
    out_cg = iter_lin(trajectory_cg.T, minima_cg)
    out_dl = iter_lin(trajectory_dl.T, minima_dl)
    out_cp = iter_lin(trajectory_cp.T, minima_cp)

    return out_cg, out_dl, out_cp

# Calculate the order Q value
# Reference: https://caam37830.github.io/book/01_analysis/convergence.html
# Q value can obtained as the slope of the equation as written in the report
def iter_q(a, minima):
    b = np.repeat(minima, repeats=a.shape[1], axis=1)
    a = a - b
    
    a = np.linalg.norm(a, axis=0)
    c = list(np.abs(np.diff(np.log(a))))
    c = np.array(list(map(lambda x: x if x > 1e-5 else 1e-3, c)))
    e = np.log(c)

    k = np.arange(len(e))

    line = np.polyfit(k, e, 1)
    q = np.exp(line[0])
    return q

# Obtain the order Q for the find out the convergence rates
def check_convergence(trajectory_cg, trajectory_dl, trajectory_cp,n):
    minima_cg = np.ones((n,1))
    minima_dl = np.ones((n,1))
    minima_cp = np.ones((n,1))

    q_cg = iter_q(trajectory_cg.T, minima_cg)
    q_dl = iter_q(trajectory_dl.T, minima_dl)
    q_cp = iter_q(trajectory_cp.T, minima_cp)

    return q_cg, q_dl, q_cp