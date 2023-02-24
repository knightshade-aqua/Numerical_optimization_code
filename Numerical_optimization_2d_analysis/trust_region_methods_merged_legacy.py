import numpy as np
import yaml
import objective_functions
from cg_steihaug import CG_Steihaug
from  trust_region_dog_leg_method import Dog_Leg
import time
from visualize_multiple_plots import plot_3d_with_trajectory, visualize_3d, visualize_contour, plot_radius
from trustregion import TrustRegion
from utils import *

""" trust_region_methods_merged.py is the main calling function for sub-problem solution methods cauchy point,
    conjugate gradient steihaug, Dog leg method """


def trust_region_cauchy(objective, init_state):

     # Create the trust region object
     # objective refers to the name of the objective function
    tr = TrustRegion(objective)

    # Initialize hyperparameters
    x0 = init_state
    delta0 = 1
    delta_max = 5
    eta1 = 0.1

    # Obtain the values of trust radius and iterations
    res, f_res, iterations = tr.run(x0, delta0, delta_max, eta1)
    traj = np.array(tr.path)

    return traj, tr.rho, iterations

# Booth fuction is used to obtain the objective function values
# Not to be confused with objective function booth
def booth(x,y, objective):

    if objective == "rosenbrock":
        f = objective_functions.rosenbrock([x, y])
    elif objective == "booth":
        f = objective_functions.booth([x, y])
    else:
        f = objective_functions.ackley([x, y])
    return f

def main():

    trajectories_cg = [] 
    trust_radius_cg = []
    trajectories_dl = []
    trust_radius_dl = []
    trajectories_cp = []

    # Define the total number of trials to be carried out
    total_iterations = 100

    # Initial variables to hold the number of decision variables, number of iterations and average run-time
    iterations_cg = np.zeros((2,total_iterations))
    iterations_cp = np.zeros((2,total_iterations))
    iterations_dl = np.zeros((2,total_iterations))
    seed_value = np.arange(total_iterations)

    # Access yaml file
    with open("parameters.yaml") as m:
        data = yaml.load(m, Loader=yaml.FullLoader)
    objective = data['objective']

    for i in range(total_iterations):

        print(f"########## Iteration : {i+1} ##########")

        # Set initial values
        np.random.seed(seed_value[i])
        x_init = np.random.randint(1,10)
        y_init = np.random.randint(1,10)
        
        init_state = np.array([x_init, y_init])

        # cg stands for conjugate gradient steihaug method
        # dl stands for Dog-leg method   
        cg = CG_Steihaug()
        dl = Dog_Leg()

        #Steihaug method
        start_time_cg = time.time()
        trajectories_cg, trust_radius_cg, iterations_cg[0,i] = cg.trust_region_steihaug(init_state)
        end_time_cg = time.time()

        # Dog leg method
        start_time_dl = time.time()
        trajectories_dl, trust_radius_dl, iterations_dl[0,i] = dl.trust_region_dog_leg(init_state)
        end_time_dl = time.time()


        # Cauchy method
        start_time_cp = time.time()
        trajectories_cp, trust_radius_cp, iterations_cp[0,i] = trust_region_cauchy(objective, init_state)
        end_time_cp = time.time()

         # Calculate the run-time
        run_time_cg = end_time_cg - start_time_cg
        run_time_dl = end_time_dl - start_time_dl
        run_time_cp = end_time_cp - start_time_cp

        iterations_cg[1,i] = run_time_cg
        iterations_dl[1,i] = run_time_dl
        iterations_cp[1,i] = run_time_cp

    # Average the values over the number of trials
    iterations_cg = np.mean(iterations_cg, axis=1)
    iterations_cp = np.mean(iterations_cp, axis=1)
    iterations_dl = np.mean(iterations_dl, axis=1)

    print("########################################################")
    print(f"The objective function is {objective}")
    print(f"The number of iterations for steihaug: {iterations_cg[0]}")
    print(f"The number of iterations for dog leg: {iterations_dl[0]}")
    print(f"The number of iterations for cauchy point: {iterations_cp[0]}")

    print(f"The value of xk for steihaug: {trajectories_cg[-1,:]}")
    print(f"The value of xk for dog leg: {trajectories_dl[-1,:]}")
    print(f"The value of xk for cauchy point: {trajectories_cp[-1,:]}")

    print(f'Execution time for steihaug method : {iterations_cg[1]}')
    print(f'Execution time for dog leg method : {iterations_dl[1]}')
    print(f'Execution time for cauchy point  method : {iterations_cp[1]}')


    # Used when plotting
    #visualize_3d(objective)
    #visualize_contour(trajectories_cg, trust_radius_cg, trajectories_dl, trust_radius_dl, trajectories_cp, objective)
    #plot_radius(trajectories_cg, trust_radius_cg, trajectories_dl, trust_radius_dl, trajectories_cp, trust_radius_cp, objective)
    #plot_3d_with_trajectory(trajectories_cg, trajectories_dl, trajectories_cp, objective)

if __name__ == "__main__":
    main()