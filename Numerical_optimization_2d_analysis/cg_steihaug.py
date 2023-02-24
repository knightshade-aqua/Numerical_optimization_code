import numpy as np
import objective_function_grad
import yaml
from subproblem_solution import subproblem_steihaug
import objective_functions
from hessian_func import hessian

class CG_Steihaug():
    def __init__(self) -> None:
        pass

    def quadratic_approximation(self, x_k, p_k, objective):

        if objective == "rosenbrock":
            f_k = objective_functions.rosenbrock(x_k)
            g_k = objective_function_grad.rosenbrock(x_k)
            h_k = hessian(x_k, objective)
        elif objective == "booth":
            f_k = objective_functions.booth(x_k)
            g_k = objective_function_grad.booth(x_k)
            h_k = hessian(x_k, objective)
        else:
            f_k = objective_functions.ackley(x_k)
            g_k = objective_function_grad.ackley(x_k)
            h_k = hessian(x_k, objective)
        return f_k + (g_k.T @ p_k) + 0.5 * (p_k.T @ h_k @ p_k)
 
    def trust_region_steihaug(self, init_state):
        
        # Load the hyper parameters
        with open("parameters.yaml") as m:
            data = yaml.load(m, Loader=yaml.FullLoader)
        eta1 = data['eta1']
        eta2 = data['eta2']
        eta3 = data['eta3']
        c1 = data['c1']
        objective = data['objective']
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
            p_k = subproblem_steihaug(x_k, delta_old, objective)
 

            # Calculate the trustworthiness of the obtained model
            if objective == "rosenbrock":
                rho = (objective_functions.rosenbrock(x_k) - objective_functions.rosenbrock(x_k + p_k)) / (self.quadratic_approximation(x_k, p_k, objective) - self.quadratic_approximation(x_k + p_k, p_k, objective))
            elif objective == "booth":
                rho = (objective_functions.booth(x_k) - objective_functions.booth(x_k + p_k)) / (self.quadratic_approximation(x_k, p_k, objective) - self.quadratic_approximation(x_k + p_k, p_k, objective))
            else:
                rho = (objective_functions.ackley(x_k) - objective_functions.ackley(x_k + p_k)) / (self.quadratic_approximation(x_k, p_k, objective) - self.quadratic_approximation(x_k + p_k, p_k, objective))


            # Assess the quality of rho to modify the trust radius
            # Checking to increase the trust radius value
            # np.isclose() is used to check whether the value of the step is almost close to delta with a tolerance
            if rho < eta2:
                delta_new = delta_old * c1
            elif (rho >= eta3 and np.isclose(np.linalg.norm(p_k), delta_old, 1e-4)):
                delta_new = min(2*delta_old, delta_max)
            else:
                delta_new = delta_old


            if rho > eta1:
                x_k_new = x_k + p_k
            else:
                x_k_new = x_k


            x_k = x_k_new
            delta_old = delta_new
            x_trajectory.append(x_k)
            trust_radius.append(delta_old)
        
            i = i + 1
            if i > 10000:
                break

            # Check for termination condition
            if objective == "rosenbrock":
                if (np.linalg.norm(objective_function_grad.rosenbrock(x_k)) < 1e-4):
                    break
            elif objective == "booth":
                if (np.linalg.norm(objective_function_grad.booth(x_k)) < 1e-4):
                    break
            else:
                if (np.linalg.norm(objective_function_grad.ackley(x_k)) < 1e-4):
                    break

        
        x_trajectory = np.array(x_trajectory)

        return x_trajectory, trust_radius, i

