import numpy as np
import yaml
from subproblem_solution_methods import subproblem_steihaug, subproblem_dog_leg
import copy

class CG_Steihaug:
    def __init__(self, f_k, g_k, h_k,n):
        # Obtain the objective function value, gradient and hessian at the current iterate
        self.f_k = f_k
        self.g_k = g_k
        self.h_k = h_k
        self.n = n

    def quadratic_approximation(self, x_k, p_k):
        # Solve for the quadratic model
        f_k = self.f_k(x_k)
        g_k = self.g_k(x_k)
        h_k = self.h_k(x_k)
        return f_k + (g_k.T @ p_k) + 0.5 * (p_k.T @ h_k @ p_k)
 
    def trust_region_steihaug(self, init_state, method):

        # Load the hyper parameters
        with open("parameters.yaml") as m:
            data = yaml.load(m, Loader=yaml.FullLoader)
        eta1 = data['eta1']
        eta2 = data['eta2']
        eta3 = data['eta3']
        c1 = data['c1']
        delta_max = data['delta_max']
        trust_radius = []
        x_trajectory = []   
        x_k = init_state    
        delta_old = data['delta_init']
        x_trajectory.append(x_k)
        trust_radius.append(delta_old)
        i = 0

        while (True):
            # Solve the subproblem to obtain the the step p_k
            if method == "steihaug":
                p_k = subproblem_steihaug(x_k, delta_old, self.f_k, self.g_k, self.h_k, self.n)
            else:
                p_k = subproblem_dog_leg(x_k, delta_old, self.f_k, self.g_k, self.h_k, self.n)
        
            # Calculate the trustworthiness of the obtained model
            rho = (self.f_k(x_k) - self.f_k(x_k + p_k)) / (self.quadratic_approximation(x_k, p_k) - self.quadratic_approximation(x_k + p_k, p_k))

            # Assess the quality of rho to modify the trust radius
            if rho < eta2:
                # Reduce delta value
                delta_new = delta_old * c1
            elif (rho >= eta3 and np.isclose(np.linalg.norm(p_k), delta_old, 1e-4)):
            # Checking to increase the trust radius value
            # np.isclose() is used to check whether the value of the step is almost close to delta with a tolerance
                delta_new = min(2*delta_old, delta_max)
            else:
                # Keep the same delta value
                delta_new = delta_old

            if rho > eta1:
                # Step improved
                x_k_new = x_k + p_k
            else:
                # No change in the step
                x_k_new = x_k

            # Update iterate values
            x_k = x_k_new
            delta_old = delta_new
            x_trajectory.append(x_k)
            trust_radius.append(delta_old)
        
            i = i + 1
            if i > 200000:
                break

            # Check for termination condition
            if (np.linalg.norm(self.g_k(x_k)) < 1e-4):
                break

        return x_trajectory, trust_radius, i
