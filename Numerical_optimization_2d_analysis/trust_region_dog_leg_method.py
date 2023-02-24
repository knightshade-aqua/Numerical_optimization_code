import numpy as np
import objective_function_grad
import yaml
from dog_leg_subproblem import subproblem_dog_leg
import objective_functions
from hessian_func import hessian


class Dog_Leg():
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

    def trust_region_dog_leg(self, init_state):

        # Load the hyper parameters
        with open("parameters.yaml") as m:
            data = yaml.load(m, Loader=yaml.FullLoader)

        eta1 = data['eta1']
        eta2 = data['eta2']
        eta3 = data['eta3']
        c1 = data['c1']
        delta_max = data['delta_max']
        objective = data['objective']
        trust_radius = []
        x_trajectory = []   
        x_k = init_state    
        delta_old = data['delta_init']
        x_trajectory.append(x_k)
        trust_radius.append(delta_old)
        i = 0

        while (True):
            # Solve the subproblem to obtain the the step p_k
            p_k = subproblem_dog_leg(x_k, delta_old, objective)

            # Calculate the trustworthiness of the obtained model
            if objective == "rosenbrock":
                rho = (objective_functions.rosenbrock(x_k) - objective_functions.rosenbrock(x_k + p_k)) / (self.quadratic_approximation(x_k, p_k, objective) - self.quadratic_approximation(x_k + p_k, p_k, objective))
            elif objective == "booth":
                rho = (objective_functions.booth(x_k) - objective_functions.booth(x_k + p_k)) / (self.quadratic_approximation(x_k, p_k, objective) - self.quadratic_approximation(x_k + p_k, p_k, objective))
            else:
                rho = (objective_functions.ackley(x_k) - objective_functions.ackley(x_k + p_k)) / (self.quadratic_approximation(x_k, p_k, objective) - self.quadratic_approximation(x_k + p_k, p_k, objective))

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
                x_k_new = x_k


            x_k = x_k_new
            delta_old = delta_new
            x_trajectory.append(x_k)
            trust_radius.append(delta_old)
        
            i = i + 1
            if i > 100000:
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