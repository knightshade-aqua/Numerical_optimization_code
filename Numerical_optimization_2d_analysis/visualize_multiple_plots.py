import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import objective_functions

""" visualize_multiple_plots.py is used to plot the graphs
    It works on hard-coded values of functions, gradients and hessian 
    Reference for plot functions: https://github.com/anweshpanda/Trust_Region/blob/main/Trust%20Region.ipynb"""

global scale
scale = 1

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

# Visualizes the 3D plot of the functions
def visualize_3d(objective):
    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    x, y = np.meshgrid(x, y)

    if objective == "rosenbrock":
        f = objective_functions.rosenbrock([x, y])
    elif objective == "booth":
        f = objective_functions.booth([x, y])
    else:
        f = objective_functions.ackley([x, y])

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, f, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_title('3D plot of' + ' '+str(objective)+' ' + 'function', fontsize = 18,fontweight='bold')
    ax.set_xlabel('X',fontsize=16,fontweight='bold')
    ax.set_ylabel('Y',fontsize=16,fontweight='bold')
    plt.savefig("Images/3d_function_plot_"+objective+".png", dpi=250)
    plt.show()
    return
    
# Visiualzes the contour plot of the function
def visualize_contour(x_trajectory_cg, trust_radius_cg, trajectory_dl, trust_radius_dl, trajectory_cp, objective):
    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    x, y = np.meshgrid(x, y)

    if objective == "rosenbrock":
        f = objective_functions.rosenbrock([x, y])
        minima = np.array([1,1/scale])
    elif objective == "booth":
        f = objective_functions.booth([x,y])
        minima = np.array([1,3/scale])
    else:
        f = objective_functions.ackley([x,y])
        minima = np.array([0,0/scale])

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    cp = ax.contour(x,y, f, np.arange(2,20)**4)

    ax.plot(x_trajectory_cg[:,0],x_trajectory_cg[:,1], marker='o', label='Conjugate gradient Steihaug trajectory')
    ax.plot(trajectory_dl[:,0],trajectory_dl[:,1], marker='x',  label='Dog leg trajectory')
    ax.plot(trajectory_cp[:,0],trajectory_cp[:,1], marker='*', label='Cauchy point trajectory')

    ax.plot(x_trajectory_cg[0,0],x_trajectory_cg[0,1], color='blue', marker='o', markersize=15)
    ax.plot(trajectory_dl[0,0],trajectory_dl[0,1], color='orange',marker='x', markersize=15)
    ax.plot(trajectory_cp[0,0],trajectory_cp[0,1], color='green', marker='*', markersize=15)

    ax.plot(x_trajectory_cg[-1,0],x_trajectory_cg[-1,1], color='blue', marker='o', markersize=15)
    ax.plot(trajectory_dl[-1,0],trajectory_dl[-1,1], color='orange',marker='x', markersize=15)
    ax.plot(trajectory_cp[-1,0],trajectory_cp[-1,1], color='green',marker='*', markersize=15)
    
    ax.annotate('Minima: ['+str(minima[0])+','+str(minima[1])+']',
            xy=(minima[0], minima[1]), xycoords='data',
            xytext=(-15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom',fontsize = 18)

    ax.clabel(cp, inline=True, fontsize=10)
    ax.set_title('Contour plot for' + ' '+str(objective)+' ' + 'function, Scale = ' +str(scale), fontsize = 18,fontweight='bold')
    ax.set_xlabel('X',fontsize=16,fontweight='bold')
    ax.set_ylabel('Y',fontsize=16,fontweight='bold')
    ax.legend()
    plt.savefig("Images/2d_contour_plot_"+' '+ objective + ' ' + '_'+ str(scale) +".png", dpi=250)
    plt.show()
    
    return

# Visualize 3D plot with trajectory
def plot_3d_with_trajectory(x_trajectory_cg, trajectory_dl, trajectory_cp, objective):
    x = np.linspace(0, 10, 1000)
    y = np.linspace(0, 10, 1000)
    x, y = np.meshgrid(x, y)
    

    if objective == "rosenbrock":
        f = objective_functions.rosenbrock([x, y])
        minima = np.array([1,(1/scale)])
    elif objective == "booth":
        f = objective_functions.booth([x, y])
        minima = np.array([1,(3/scale)])
    else:
        f = objective_functions.ackley([x, y])
        minima = np.array([0,(0/scale)])
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, f, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.4)

    # Trajectory value calculation
    function_value_cg = booth(x_trajectory_cg[:,0], x_trajectory_cg[:,1], objective)
    function_value_dl = booth(trajectory_dl[:,0], trajectory_dl[:,1], objective)
    function_value_cp = booth(trajectory_cp[:,0], trajectory_cp[:,1], objective)

    # Trajectory initial points and end points
    function_value_point_begin_cg = booth(x_trajectory_cg[0,0], x_trajectory_cg[0,1], objective)
    function_value_point_end_cg = booth(x_trajectory_cg[-1,0], x_trajectory_cg[-1,1], objective)

    function_value_point_begin_dl = booth(trajectory_dl[0,0], trajectory_dl[0,1], objective)
    function_value_point_end_dl = booth(trajectory_dl[-1,0], trajectory_dl[-1,1], objective)

    function_value_point_begin_cp = booth(trajectory_cp[0,0], trajectory_cp[0,1], objective)
    function_value_point_end_cp = booth(trajectory_cp[-1,0], trajectory_cp[-1,1], objective)

    function_minima = booth(minima[0], minima[1], objective)

    ax.plot(x_trajectory_cg[:, 0], x_trajectory_cg[:, 1], zs=function_value_cg, color='red', label='Conjugate gradient Steihaug trajectory')
    ax.plot(x_trajectory_cg[0,0], x_trajectory_cg[0,1], zs=function_value_point_begin_cg, color='blue', marker='*', markersize=10)
    ax.plot(x_trajectory_cg[-1,0], x_trajectory_cg[-1,1], zs=function_value_point_end_cg, color='blue', marker='+', markersize=10)

    ax.plot(trajectory_dl[:, 0], trajectory_dl[:, 1], zs=function_value_dl, color='green', label='Dog leg trajectory')
    ax.plot(trajectory_dl[0,0], trajectory_dl[0,1], zs=function_value_point_begin_dl, color='red', marker='*', markersize=10)
    ax.plot(trajectory_dl[-1,0], trajectory_dl[-1,1], zs=function_value_point_end_dl, color='red', marker='+', markersize=10)

    ax.plot(trajectory_cp[:, 0], trajectory_cp[:, 1], zs=function_value_cp, color='blue', label='Cauchy point trajectory')
    ax.plot(trajectory_cp[0,0], trajectory_cp[0,1], zs=function_value_point_begin_cp, color='yellow', marker='*', markersize=10)
    ax.plot(trajectory_cp[-1,0], trajectory_cp[-1,1], zs=function_value_point_end_cp, color='yellow', marker='+', markersize=10)
    ax.plot(minima[0], minima[1], zs=function_minima, color='cyan', marker='o', markersize=3, label='Minima: ['+str(minima[0])+','+str(minima[1])+']')
    ax.set_title('3D Plot with trajectory for ' +objective+' function, Scale = ' +str(scale), fontsize = 18,fontweight='bold')
    ax.set_xlabel('X',fontsize=16,fontweight='bold')
    ax.set_ylabel('Y',fontsize=16,fontweight='bold')
    plt.legend()
    plt.savefig("Images/3d_function_with_trajectory_"+' '+ objective + ' ' + '_'+ str(scale) +".png", dpi=250)
    plt.show()
    
    return

# Obtain contour plots with radius
def plot_radius(x_trajectory_cg, trust_radius_cg, trajectory_dl, trust_radius_dl, trajectory_cp, trust_radius_cp, objective):
    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    x, y = np.meshgrid(x, y)

    if objective == "rosenbrock":
        f = objective_functions.rosenbrock([x, y])
        minima = np.array([1,1*scale])
    elif objective == "booth":
        f = objective_functions.booth([x,y])
        minima = np.array([1,3*scale])
    else:
        f = objective_functions.ackley([x,y])
        minima = np.array([0,0*scale])

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    cp = ax.contour(x,y, f, np.arange(2,10)**4)
    ax.plot(x_trajectory_cg[:,0],x_trajectory_cg[:,1], marker='o', label="Conjugate gradient trajectory")

    
    ax.annotate('Minima: ['+str(minima[0])+','+str(minima[1]*scale)+']',
            xy=(minima[0], minima[1]), xycoords='data',
            xytext=(-15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom',fontsize = 18)
    color = ['red','blue','green','yellow']

    for i in range(len(trust_radius_cg)):

        circle_cg = plt.Circle(x_trajectory_cg[i,:], radius=trust_radius_cg[i],facecolor = color[i%len(color)],alpha = 0.2)

        ax.add_artist(circle_cg)
    
    ax.clabel(cp, inline=True, fontsize=10)
    ax.set_title('Contour plot for Conjugate gradient Steihaug subproblem, Scale = ' +str(scale), fontsize = 18,fontweight='bold')
    ax.set_xlabel('X',fontsize=16,fontweight='bold')
    ax.set_ylabel('Y',fontsize=16,fontweight='bold')
    ax.legend()
    plt.savefig("Images/2d_contour_plot_steihaug_radius.png", dpi=250)
    plt.show()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    cp = ax.contour(x,y, f, np.arange(2,20)**4)
    ax.plot(trajectory_dl[:,0],trajectory_dl[:,1], marker='x', label="Dog leg trajectory")

    
    ax.annotate('Minima: ['+str(minima[0])+','+str(minima[1]*scale)+']',
            xy=(minima[0], minima[1]), xycoords='data',
            xytext=(-15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom',fontsize = 18)
    color = ['red','blue','green','yellow']

    for i in range(len(trust_radius_dl)):
        circle_dl = plt.Circle(trajectory_dl[i,:], radius=trust_radius_dl[i],facecolor = color[i%len(color)],alpha = 0.2)
        ax.add_artist(circle_dl)
    ax.clabel(cp, inline=True, fontsize=10)
    ax.set_title('Contour plot for Dog leg subproblem, Scale = ' +str(scale), fontsize = 18,fontweight='bold')
    ax.set_xlabel('X',fontsize=16,fontweight='bold')
    ax.set_ylabel('Y',fontsize=16,fontweight='bold')
    ax.legend()
    plt.savefig("Images/2d_contour_plot_dog_leg_radius.png", dpi=250)
    plt.show()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    cp = ax.contour(x,y, f, np.arange(2,20)**4)
    ax.plot(trajectory_cp[:,0],trajectory_cp[:,1], marker='*', label="Cauchy point trajectory")
    
    ax.annotate('Minima: ['+str(minima[0])+','+str(minima[1]*scale)+']',
            xy=(1, 3), xycoords='data',
            xytext=(-15, 25), textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='bottom',fontsize = 18)
    color = ['red','blue','green','yellow']
    for i in range(len(trust_radius_cp)):
        circle_cp = plt.Circle(trajectory_cp[i,:], radius=trust_radius_cp[i],facecolor = color[i%len(color)],alpha = 0.2)
        ax.add_artist(circle_cp)
    ax.clabel(cp, inline=True, fontsize=10)
    ax.set_title('Contour plot for Cauchy point subproblem, Scale = ' +str(scale), fontsize = 18,fontweight='bold')
    ax.set_xlabel('X',fontsize=16,fontweight='bold')
    ax.set_ylabel('Y',fontsize=16,fontweight='bold')
    ax.legend()
    plt.savefig("Images/2d_contour_plot_cauchy_radius.png", dpi=250)
    plt.show()


    return