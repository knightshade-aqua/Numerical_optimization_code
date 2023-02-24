import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy import linalg as la


def plot_contour_with_trajectories(f, traj):
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)
    z = np.empty((len(y), len(x)))

    for i in range(len(x)):
        for j in range(len(y)):
            z[j, i] = f(np.array([x[i], y[j]]))

    plt.contourf(x, y, z, 15, cmap=cm.viridis)
    plt.colorbar()

    plt.plot(traj[:, 0], traj[:, 1], '-b', label='Trajectory')
    plt.plot(traj[0, 0], traj[0, 1], 'xy', label='Starting point')
    plt.plot(traj[-1, 0], traj[-1, 1], 'og', label='End point')
    plt.legend()
    plt.show()


def plot_3d_objective(f):
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)
    x, y = np.meshgrid(x, y)
    z = f(np.array([x, y]))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.viridis, linewidth=0, antialiased=True)
    plt.show()


def plot_3d_with_trajectories(f, traj):
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)
    x, y = np.meshgrid(x, y)
    z = f(np.array([x, y]))

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.4)

    f_traj = f(np.array([traj[:, 0], traj[:, 1]]))
    ax.plot(traj[:, 0], traj[:, 1], zs=f_traj, color='red', label='trajectory')
    plt.show()


# Reference: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
# Used to obtain the nearest positive definte hessian
def nearest_pd(a):
    b = (a + a.T) / 2
    _, s, v = la.svd(b)

    h = v.T @ np.diag(s) @ v

    a2 = (b + h) / 2
    a3 = (a2 + a2.T) / 2

    if is_pd(a3):
        return a3

    spacing = np.spacing(la.norm(a))

    identity = np.eye(a.shape[0])
    k = 1
    while not is_pd(a3):
        min_eig = np.min(np.real(la.eigvals(a3)))
        a3 += identity * (-min_eig * k**2 + spacing)
        k += 1

    return a3


def is_pd(b):
    return np.all(la.eigvals(b) > 0)


def is_psd(b):
    return np.all(la.eigvals(b) >= 0)
