import numpy as np


""" objective_functions.py gives the hard-coded objective function values """

global scale
scale = 1

def rosenbrock(state):
    
    # Return the rosenbrock function and its gradient for the given ranges of x and y
    x = state[0]
    y = state[1]*scale
    f = (1 - x)**2 + 100 * (y - x**2)**2    # value of the function
    return f

def ackley(state):

    # Return the ackley function and its gradient for the given ranges of x and y
    x = state[0]
    y = state[1]*scale
    f = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.exp(1) + 20    # value of the function
    return f

def booth(state):

    x = state[0]
    y = state[1]*scale
    # Return the booth function and its gradient for the given ranges of x and y
    f = (x + 2 * y - 7)**2 + (2 * x + y - 5)**2    # value of the function
    return f

