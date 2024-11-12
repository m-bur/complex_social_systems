import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def polarization(network):
    """
    Returns the average opinion of  the network.
    """
    s = 0 # stores the sum of all opinions
    n = np.shape(network)
    for i in range(n[0]):
        for j in range(n[1]):
            s += network[i][j].get_opinion()
    return s / (n[0]*n[1]) # returns the average opinon for the network


def local_clustering(voter, network):
    """
    Returns the local clustering (i.e. the fraction of agreement between one voter and its neighbors)
    """
    c = 0 # counter
    n = voter.get_number_of_neighbors() # number of neighbours of voter
    my_opinion = voter.get_opinion() # voter's opinion
    my_neighbors = voter.get_neighbors() #list of neighbor coordinates
    for coordinate in my_neighbors:
        if my_opinion == network[coordinate[0]][coordinate[1]].get_opinion(): #compare voters opinion with neighbor's opinion
            c += 1
    return c / n
        

def clustering(network):
    """
    Return the clustering (degree of agreement between neighbours)
    """
    clust = 0
    n = np.shape(network)
    for i in range(n[0]):
        for j in range(n[1]):
            voter = network[i][j]
            loc_clust = local_clustering(voter, network)
            clust += loc_clust
    return clust / (n[0] * n[1])


def neighbor_opinion(voter, network):
    """
    Return the average opinion of the neighbors of the voter
    """
    s = 0
    my_neighbors = voter.get_neighbors() #list of neighbor coordinates
    n = len(my_neighbors)
    for coordinate in my_neighbors:
        s+=network[coordinate[0]][coordinate[1]].get_opinion() #compare voters opinion with neighbor's opinion
    return s / n


def neighbor_opinion_distribution(network):
    """
    Displays the distribution of the average opinion of neighbors depending on the opinon of the voter. Returns the the mean and standard deviation for each catagory.
    """
    n = np.shape(network)
    neigh_opinion_dist = {-1: [], 0: [], 1: []}
    for i in range(n[0]):
        for j in range(n[1]):
            voter = network[i][j]
            opinion = voter.get_opinion()
            neigh_opinion = neighbor_opinion(voter, network)
            np.append(neigh_opinion_dist[opinion], neigh_opinion)
    for i in neigh_opinion_dist:
        avg = np.average(neigh_opinion_dist[i])
        std = np.std(neigh_opinion_dist[i])
        plt.hist(neigh_opinion_dist[i],label=f"Opinion {i}: ave = {avg}, std = {std}")
    plt.legend()
    plt.xlabel('xi')
    plt.ylabel('Frequency of xi')
    plt.savefig('figures/neighbor_distribution.pdf')


def power_law(x, a, b):
    """
    Fit function for degree distribution. For y scale free network 2 < b < 3.
    """
    return a * np.power(x, b)

def deg_distribution(network):
    """
    Fit the degree distribution of the network to a power law and plot the results. Returns the exponent of the power law fit.
    """
    k = np.array([]) # list of the degrees of each node
    n = np.shape(network)
    for i in range(n[0]):
        for j in range(n[1]):
            k = np.append(k,network[i][j].get_number_of_neighbors()) # add the nuber of neighbors for each node to k
    unique_elements, counts = np.unique(k, return_counts=True) # count unique elements in k
    params, covariance = curve_fit(power_law, unique_elements, counts) # fit degree distribution to power law
    a, b = params
    #plot the results
    xvals = np.linspace(min(unique_elements),max(unique_elements),num=100)
    yvals = power_law(xvals, a, b)
    plt.plot(xvals, yvals, 'k--', label=f"exponent b = {b}")
    plt.plot(unique_elements, counts, 'ko', label="degree distribution")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.legend()
    plt.savefig('figures/deg_distribution.pdf')
    return b