from utils import *

class TrustRegion:
    def __init__(self, f_k, g_k, h_k):
        # Obtain the objective function value, gradient and hessian at the current iterate
        self.f = f_k
        self.g = g_k
        self.h = h_k

        # self.rho stands for trust radius
        self.rho = []

        # self.path stands for trajectory
        self.path = []

    def quadratic_model(self, x, p, b):
        # Solve for the quadratic model
        return self.f(x) + self.g(x).T @ p + 0.5 * p.T @ b @ p

    def cauchy_point(self, x, delta):
        gx = self.g(x) # gradient
        b = self.h(x) # hessian

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
        pc = -(tau * delta / (g_norm)) * gx
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

            # Calculate the trustworthiness of the obtained iterate
            rho = (self.f(x) - self.f(x+p)) / (self.quadratic_model(x, p, b) - self.quadratic_model(x+p, p, b))

            # Append the trajectory and rho values
            self.rho.append(rho)
            self.path.append(x)

            # Checking for modification conditions of the radius based on trustworthiness of the model
            if rho < eta2:
                # Reduce delta value
                delta = eta1 * delta

            # Checking to increase the trust radius value
            # np.isclose() is used to check whether the value of the step is almost close to delta with a tolerance
            elif rho >= eta3 and np.isclose(la.norm(p), delta, 1e-4):
                delta = min(2 * delta, delta_max)

            if rho > eta1:
                # Step is modified
                x = x + p
            elif (np.linalg.norm(self.g(x)) < 1e-4):
                # Checking for termination condition
                break
            if i > 200000:
                # Discontinue the loop if the iteartions exceed a certain threshold
                break

        return x, self.f(x), i, self.rho
