import numpy as np
from src.constants import c_const as co
import math
import matplotlib.pyplot as plt

def generalized_legendre_polynom_all_m(l, N, resolution):

    theta_array = np.linspace(0, np.pi, resolution+1, endpoint=False)[1:]
    #starting values
    legendre_polynom = np.empty((2*l+1, len(theta_array)), dtype=np.float64)

    if l == 0:
        legendre_polynom[0] = np.ones_like(theta_array)
        return legendre_polynom
    if abs(N) > l:
        legendre_polynom[:] = np.zeros_like(legendre_polynom)
        return legendre_polynom

        
    legendre_polynom[2*l] = (-1)**(l+N)*(math.factorial(2*l)/(math.factorial(l+N)*math.factorial(l-N)))**(1/2)*np.sin(1/2*theta_array)**(l-N)*np.cos(1/2*theta_array)**(l+N)
    legendre_polynom[2*l-1] = np.sqrt(2*l)*1/np.sin(theta_array)*(N/l-np.cos(theta_array))*legendre_polynom[2*l]
    legendre_polynom[0] = (math.factorial(2*l)/(math.factorial(l+N)*math.factorial(l-N)))**(1/2)*np.sin(1/2*theta_array)**(l+N)*np.cos(1/2*theta_array)**(l-N)
    legendre_polynom[1] = np.sqrt(2*l)*1/np.sin(theta_array)*(N/l+np.cos(theta_array))*legendre_polynom[0]

    for theta_i, theta in enumerate(theta_array):
        meeting_point = round(N*np.cos(theta))
        for i in range(2*l-2,1,-1):
            m = i-l
            if meeting_point <= m:
                m = i-l+1
                legendre_polynom[i, theta_i] = 1 / co(l, m) * (2 * (N / np.sin(theta) - m * np.cos(theta) / np.sin(theta)) * legendre_polynom[i+1, theta_i] - co(l, m+1) * legendre_polynom[i+2, theta_i])
            else:
                break
            
        for i in range(2,2*l-1,+1):
            m = i-l
            if meeting_point > m:
                m = i-l-1
                legendre_polynom[i, theta_i] = 1 / co(l, m+1) * (2 * (N / np.sin(theta) - m * np.cos(theta) / np.sin(theta)) * legendre_polynom[i-1, theta_i] - co(l, m) * legendre_polynom[i-2, theta_i])
            else:
                break

    return legendre_polynom
        
        
def generalized_legendre_polynom(l, N, m, resolution):
    if abs(m) > l:
        theta_array = np.linspace(0, np.pi, resolution + 1, endpoint=False)[1:]
        legendre = np.zeros_like(theta_array)
        return legendre

    legendre = generalized_legendre_polynom_all_m(l, N, resolution)
    m_index = m + l
    return legendre[m_index]


def plot_single_glp(l, N, m, resolution):
    theta_array = np.linspace(0, np.pi, resolution + 1, endpoint=False)
    theta_array = theta_array[1:]

    plt.figure(figsize=(8, 6))
    plt.title(f'Generalised Legendre Function l=${l}$, N=${N}$, m=${m}$', fontsize=16)
    plt.xlabel('$θ$', fontsize=14)
    plt.ylabel('$P_{l}^{N,m}$', fontsize=14)
    plt.plot(theta_array, generalized_legendre_polynom(l, N, m, resolution), label=f'P$_{{{l}}}^{{{N},{m}}}$')
    plt.legend(loc='upper center', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlim(0, np.pi)
    # plt.ylim(-1.1,1.1)
    plt.show()


def plot_all_m_glp(l, N, resolution):
    theta_array = np.linspace(0, np.pi, resolution + 1, endpoint=False)
    theta_array = theta_array[1:]

    legendre = generalized_legendre_polynom_all_m(l, N, resolution)
    plt.figure(figsize=(8, 7))
    plt.title(f'Generalised Legendre Functions l=${l}$, N=${N}$', fontsize=16)
    plt.xlabel('$θ$', fontsize=14)
    plt.ylabel('$P_l^{N,m}$', fontsize=14)
    for i in range(2 * l, -1, -1):
        plt.plot(theta_array, legendre[i], label=f'P$_{{{l}}}^{{{N},{i - l}}}$')
    ncol = math.ceil((2 * l + 1) / 2)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=14, ncol=ncol)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlim(0, np.pi)
    plt.ylim(-1.1, 1.1)
    plt.show()
