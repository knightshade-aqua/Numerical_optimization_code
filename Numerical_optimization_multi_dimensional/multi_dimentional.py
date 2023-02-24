import casadi as ca


def values(n):
    
    # Obtain the casadi variables
    x = ca.SX.sym('x', n)
    k = 0

    # Formulate the Rosenbrock function
    for i in range(n-1):
        k = k + (100*((x[i+1] - x[i]**2)**2) + (1 - x[i])**2)

    # Obtain the objective function, gradient and hessian place holders
    f = ca.Function('t',[x],[k])
    h_t, g_t = ca.hessian(k,x)
    h = ca.Function('hessian',[x],[h_t])
    g = ca.Function('gradient', [x], [g_t])

    return f, g, h

