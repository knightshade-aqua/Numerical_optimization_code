import numpy as np
import matplotlib.pyplot as plt
from cg_multi import CG_Steihaug
import time
from convergence import calculate_convergence_quad, calculate_convergence_lin, check_convergence
from multi_dimentional import *
from convergence import check_convergence
from trustregion import TrustRegion
from utils import *

""" modular.py is the main calling function for sub-problem solution methods cauchy point,
    conjugate gradient steihaug, Dog leg method """
     
def trust_region_cauchy(init_state, f_k, g_k, h_k):

    # Create the trust region object
    tr = TrustRegion(f_k, g_k, h_k)

    # Initialize hyperparameters
    delta0 = 1
    delta_max = 5
    eta1 = 0.1

    # Obtain the values of trust radius and iterations
    res, f_res, iterations, trust_radius_cp = tr.run(init_state, delta0, delta_max, eta1)
    traj = tr.path

    return traj, tr.rho, iterations, trust_radius_cp

def main():

    # Number of decision variables
    n = [10,20,30,40,50,60,70,80,90,100]
    #n = [10,20]
    #n = [10,20,30,40,50]

    # Define the total number of trials to be carried out
    total_iterations = 100

    # Initial variables to hold the number of decision variables, number of iterations and average run-time
    iterations_cg = np.zeros((3,len(n), total_iterations))
    iterations_dl = np.zeros((3,len(n), total_iterations))
    iterations_cp = np.zeros((3,len(n), total_iterations))

    # Initial variables to hold the order value Q for convergence analysis
    q_cg = np.zeros((2, len(n), total_iterations))
    q_dl = np.zeros((2, len(n), total_iterations))
    q_cp = np.zeros((2, len(n), total_iterations))

    # Initialize variables to hold the constant C values obtained according to the convergence rate formulas (Quadratic case)
    quad_cg = np.zeros((2, len(n), total_iterations))
    quad_cp = np.zeros((2, len(n), total_iterations))
    quad_dl = np.zeros((2, len(n), total_iterations))

    #Initialize variables to hold the constant C values obtained according to the convergence rate formulas (linear case)
    lin_cg = np.zeros((2, len(n), total_iterations))
    lin_cp = np.zeros((2, len(n), total_iterations))
    lin_dl = np.zeros((2, len(n), total_iterations))

    # Create an array of seed values to use for the number of trials
    seed_value = np.arange(total_iterations)

    for j in range(total_iterations):

        print(f"##### The total iteration is : {j+1} ##### ")

        for i in range(len(n)):

            print(f"The number of decision variables is : {n[i]}")

            # Store value of number of decision variables
            iterations_cg[0,i,j] = n[i]
            iterations_dl[0,i,j] = n[i]
            iterations_cp[0,i,j] = n[i]

            q_cg[0,i,j] = n[i]
            q_cp[0,i,j] = n[i]
            q_dl[0,i,j] = n[i]

            quad_cg[0,i,j] = n[i]
            quad_dl[0,i,j] = n[i]
            quad_cp[0,i,j] = n[i]
            
            lin_cg[0,i,j] = n[i]
            lin_dl[0,i,j] = n[i]
            lin_cp[0,i,j] = n[i]
        
            # Obtain the place holder for objective function, gradient, exact hessian implemented in Casadi
            f_k, g_k, h_k = values(n[i])

            # Generate initial state to be given to the algorithm
            np.random.seed(seed_value[j])
            init_state = np.random.randint(10, size=n[i])
            init_state = np.expand_dims(init_state, axis=1)

            # Create class objects run the sub-problem solution variants
            # cg stands for conjugate gradient steihaug method
            # dl stands for Dog-leg method
            # Both methods have the same class called as it contains the sub problem to solve both the variants
            cg = CG_Steihaug(f_k, g_k, h_k, n[i])
            dl = CG_Steihaug(f_k, g_k, h_k, n[i])

            # Obtain iterations, trajectories and run time for Cauchy method
            start_time_cp = time.time()
            trajectories_cp, trust, iterations_cp[1,i,j], trust_radius_cp  = trust_region_cauchy(init_state, f_k, g_k, h_k)
            end_time_cp = time.time()

            # Obtain iterations, trajectories and run time for Steihaug method. The "method" specifies the method to be used
            start_time_cg = time.time()
            trajectories_cg, trust_radius_cg, iterations_cg[1,i,j] = cg.trust_region_steihaug(init_state, method="steihaug")
            end_time_cg = time.time()

            # Obtain iterations, trajectories and run time for Dog-leg method
            start_time_dl = time.time()
            trajectories_dl, trust_radius_dl, iterations_dl[1,i,j] = dl.trust_region_steihaug(init_state, method="dog_leg")
            end_time_dl = time.time()

            # Format the trajectories to perform convergence analysis
            trajectories_cg = np.squeeze(np.array(trajectories_cg))
            trajectories_cp = np.squeeze(np.array(trajectories_cp))
            trajectories_dl = np.squeeze(np.array(trajectories_dl))

            # Obtain the order value Q for the convergence rate analysis for the methods
            q_cg[1,i,j], q_dl[1,i,j], q_cp[1,i,j] = check_convergence(trajectories_cg, trajectories_dl, trajectories_cp, n[i])
            
            # Obtain the constant values 'C' with regard to the convergence formulas for quadratic convergence
            quad_cg[1,i,j], quad_dl[1,i,j], quad_cp[1,i,j] = calculate_convergence_quad(trajectories_cg, trajectories_dl, trajectories_cp, n[i])
            
            # Obtain the constant values 'C' with regard to the convergence formulas for linear convergence
            lin_cg[1,i,j], lin_dl[1,i,j], lin_cp[1,i,j] = calculate_convergence_lin(trajectories_cg, trajectories_dl, trajectories_cp, n[i])
            
            # Calculate the run-time
            run_time_cg = end_time_cg - start_time_cg
            run_time_dl = end_time_dl - start_time_dl
            run_time_cp = end_time_cp - start_time_cp

            iterations_cg[2,i,j] = run_time_cg
            iterations_dl[2,i,j] = run_time_dl
            iterations_cp[2,i,j] = run_time_cp

    # Average the values over the number of trials
    mean_q_cg = np.mean(q_cg, axis=2)
    mean_q_cp = np.mean(q_cp, axis=2)
    mean_q_dl = np.mean(q_dl, axis=2)

    mean_quad_cg = np.mean(quad_cg, axis=2)
    mean_quad_cp = np.mean(quad_cp, axis=2)
    mean_quad_dl = np.mean(quad_dl, axis=2)

    mean_lin_cg = np.mean(lin_cg, axis=2)
    mean_lin_cp = np.mean(lin_cp, axis=2)
    mean_lin_dl = np.mean(lin_dl, axis=2)

    mean_iterations_cg = np.mean(iterations_cg, axis=2)
    mean_iterations_cp = np.mean(iterations_cp, axis=2)
    mean_iterations_dl = np.mean(iterations_dl, axis=2)

    # Printing the results
    print(f"The iterations of Steihaug method is : {mean_iterations_cg[1,:]}")
    print(f"The iterations of Dog leg method is : {mean_iterations_dl[1,:]}")
    print(f"The iterations of Steihaug method is : {mean_iterations_cp[1,:]}")
    
    print(f"The value of q for Steihaug : {mean_q_cg}")
    print(f"The value of q for Dogleg : {mean_q_cp}")
    print(f"The value of q for Cauchy point : {mean_q_dl}")

    print(f"The value of quad C for Steihaug : {mean_quad_cg}")
    print(f"The value of quad C for Dog leg : {mean_quad_dl}")
    print(f"The value of quad C for Cauchy : {mean_quad_cp}")

    print(f"The value of lin C for Steihaug : {mean_lin_cg}")
    print(f"The value of lin C for Dog leg : {mean_lin_dl}")
    print(f"The value of lin C for Cauchy : {mean_lin_cp}")
 
    # Plot for number of iterations v/s number of decision variables
    fig = plt.figure(figsize=(10,10))
    plt.plot(mean_iterations_cg[0,:], mean_iterations_cg[1,:], linestyle='dashed', marker="o", label="Conjugate gradient  Steihaug iterations")
    plt.plot(mean_iterations_cp[0,:], mean_iterations_cp[1,:], linestyle='dashed', marker="x", label="Cauchy point iterations")
    plt.plot(mean_iterations_dl[0,:], mean_iterations_dl[1,:], linestyle='dashed',marker="*", label="Dog leg iterations")
    plt.xlabel('Number of variables',fontsize=16,fontweight='bold')
    plt.ylabel('Number of iterations',fontsize=16,fontweight='bold')
    plt.title("Progression of iterations v/s number of variables", fontsize = 14,fontweight='bold')
    plt.legend()
    plt.savefig("Images/multidimentinal_iterations.png", dpi=250)

    # Plot for number of run-time v/s number of decision variables
    fig = plt.figure(figsize=(10,10))
    plt.plot(mean_iterations_cg[0,:], mean_iterations_cg[2,:], linestyle='dashed',marker="o", label="Conjugate gradient  Steihaug iterations")
    plt.plot(mean_iterations_cp[0,:], mean_iterations_cp[2,:], linestyle='dashed',marker="x", label="Cauchy point iterations")
    plt.plot(mean_iterations_dl[0,:], mean_iterations_dl[2,:], linestyle='dashed',marker="*", label="Dog leg iterations")
    plt.xlabel('Number of variables',fontsize=16,fontweight='bold')
    plt.ylabel('Run time (s)',fontsize=16,fontweight='bold')
    plt.legend()
    plt.title("Progression of run time v/s number of variables", fontsize = 14,fontweight='bold')
    plt.savefig("Images/multidimentinal_run_time.png", dpi=250)

if __name__ == "__main__":
    main()