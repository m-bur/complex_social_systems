import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def power_law(x, a, b):
    """
    fit function for degree distribution. For y scale free network 2 < b < 3
    """
    return a * np.power(x, b)


def deg_distribution(network):
    """
    fit the degree distribution of the network to a power law and plot the results
    """
    k = np.array([]) # list of the degrees of each node
    n = np.shape(network)
    for i in range(n[0]):
        for j in range(n[1]):
            k = np.append(k,network[i][j].get_number_of_neighbors())  # add the nuber of neighbors for each node to k
    unique_elements, counts = np.unique(k, return_counts=True)  # count unique elements in k
    params, covariance = curve_fit(power_law, unique_elements, counts)  # fit degree distribution to power law
    a, b = params
    # plot the results
    xvals = np.linspace(min(unique_elements),max(unique_elements),num=100)
    yvals = power_law(xvals, a, b)
    plt.plot(xvals, yvals, 'k--', label=f"exponent b = {b}")
    plt.plot(unique_elements, counts, 'ko', label="degree distribution")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.legend()
    plt.savefig('figures/deg_distribution.pdf')
    return b

