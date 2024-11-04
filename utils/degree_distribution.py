import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def power_law(x, a, b):
    return a * np.power(x, b)

def deg_distribution(network):
    k = np.array([])
    n = np.shape(network)
    for i in range(n[0]):
        for j in range(n[1]):
            k = np.append(k,network[i][j].get_number_of_neighbors())
    unique_elements, counts = np.unique(k, return_counts=True)
    params, covariance = curve_fit(power_law, unique_elements, counts)
    a, b = params
    xvals = np.linspace(min(unique_elements),max(unique_elements),num=100)
    yvals = power_law(xvals, a, b)
    plt.plot(xvals, yvals, 'k--', label=f"exponent b = {b}")
    plt.plot(unique_elements, counts, 'ko', label="degree distribution")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.legend()
    plt.savefig('figures/deg_distribution.pdf')
    return b
