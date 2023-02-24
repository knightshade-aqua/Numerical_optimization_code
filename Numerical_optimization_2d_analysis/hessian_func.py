import numpy as np

""" hessian_func.py is used to obtain the hessian values for the objective function.
The hessian values are hard-coded """

def hessian(state, objective):
    scale = 1
    x = state[0]
    y = state[1]*scale

    if objective == "rosenbrock":
        H = np.array([[-400*(y - 3*x**2) + 2, -400*x],
                     [-400*x, 200]])
        return H
    elif objective == "booth":
        H = np.array([[10, 8],
            [8, 10]])
        return H
    else:
        fxx = 0.0565685424949238*x**2*np.exp(-0.14142135623731*np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2) - 0.5*np.pi**2*np.exp(0.5*np.cos(2*np.pi*x) + 0.5*np.cos(2*np.pi*y))*np.sin(2*np.pi*x)**2 + 1.0*np.pi**2*np.exp(0.5*np.cos(2*np.pi*x) + 0.5*np.cos(2*np.pi*y))*np.cos(2*np.pi*x) - 0.4*np.exp(-0.14142135623731*np.sqrt(x**2 + y**2))
        fxy = 0.0565685424949238*x*y*np.exp(-0.14142135623731*np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2) - 0.5*np.pi**2*np.exp(0.5*np.cos(2*np.pi*x) + 0.5*np.cos(2*np.pi*y))*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
        fyx = 0.0565685424949238*x*y*np.exp(-0.14142135623731*np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2) - 0.5*np.pi**2*np.exp(0.5*np.cos(2*np.pi*x) + 0.5*np.cos(2*np.pi*y))*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
        fyy = 0.0565685424949238*y**2*np.exp(-0.14142135623731*np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2) - 0.5*np.pi**2*np.exp(0.5*np.cos(2*np.pi*x) + 0.5*np.cos(2*np.pi*y))*np.sin(2*np.pi*y)**2 + 1.0*np.pi**2*np.exp(0.5*np.cos(2*np.pi*x) + 0.5*np.cos(2*np.pi*y))*np.cos(2*np.pi*y) - 0.4*np.exp(-0.14142135623731*np.sqrt(x**2 + y**2))

        H = np.array([[fxx, fxy],
                    [fyx, fyy]])
        return H
