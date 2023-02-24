from utils import *
import objective_functions
import objective_function_grad
from hessian_func import hessian


class TrustRegion:
    def __init__(self, objective):
        # Obtain the objective, gradient and hessian function
        if objective == "rosenbrock":
            self.f = lambda x_k: objective_functions.rosenbrock(x_k)
            self.g = lambda x_k: objective_function_grad.rosenbrock(x_k)
            self.h = lambda x_k: hessian(x_k, objective)
        elif objective == "booth":
            self.f = lambda x_k: objective_functions.booth(x_k)
            self.g = lambda x_k: objective_function_grad.booth(x_k)
            self.h = lambda x_k: hessian(x_k, objective)
        else:
            self.f = lambda x_k: objective_functions.ackley(x_k)
            self.g = lambda x_k: objective_function_grad.ackley(x_k)
            self.h = lambda x_k: hessian(x_k, objective)

        # self.rho stands for trust radius
        self.rho = []

        # self.path stands for trajectory
        self.path = []

    def quadratic_model(self, x, p, b):
        # Solve for the quadratic model
        return self.f(x) + self.g(x).T @ p + 0.5 * p.T @ b @ p

    def cauchy_point(self, x, delta):
        gx = self.g(x)# gradient
        b = self.h(x)# hessian

        # Check if 'b' is Positive Semi Definite
        if not is_psd(b):
            b = nearest_pd(b)

        gt_bk_g = gx.T @ b @ gx
        g_norm = la.norm(gx)


        # Find value of tau first
        if gt_bk_g <= 0:
            tau = 1
        else:
            tau = min(g_norm**3 / (delta * gt_bk_g), 1)

        # Calculate the cauchy point and normalize
        pc = -(tau * delta / g_norm) * gx
        
        mul = np.floor(delta / la.norm(pc).astype('f')) * (1 - 1e-3)
        if mul == 0:
            return pc * (1 - 1e-3), b
        else:
            return mul * pc, b

    def run(self, x0, delta0, delta_max, eta1, eta2=0.25, eta3=0.75):
        x = x0 # Initial value
        delta = delta0 # Initial trust radius value
        i = 0

        while True:
            i += 1

            # Obtain step (p) and hessian (b) by solving for cauchy point
            p, b = self.cauchy_point(x, delta)

             # Calculate the trustworthiness of the obtained model
            rho = (self.f(x) - self.f(x+p)) / (self.quadratic_model(x, p, b) - self.quadratic_model(x+p, p, b))

            # Append the trajectory and rho values
            self.rho.append(rho)
            self.path.append(x)
            
            # Checking to increase the trust radius value
            # np.isclose() is used to check whether the value of the step is almost close to delta with a tolerance
            if rho < eta2:
                delta = eta1 * delta
            elif (rho >= eta3 and np.linalg.norm(p) == delta):
                delta = min(2 * delta, delta_max)

            if rho > eta1:
                # Step is modified
                x = x + p
            elif (np.linalg.norm(self.g(x)) < 1e-4):
                # Checking for termination condition
                break
            if i > 100000:
                # Discontinue the loop if the iteartions exceed a certain threshold
                break
        
        # i corresponds to iterates
        return x, self.f(x), i
