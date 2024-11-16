import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def opinion_list(network):
    """
    Returns the list of all opinions
    """
    opinions = []
    n = np.shape(network)
    for i in range(n[0]):
        for j in range(n[1]):
            opinions.append(network[i][j].get_opinion())
    return opinions

def polarization(network):
    """
    Returns the average opinion of  the network.
    """
    return np.mean(opinion_list(network))

def std_opinion(network):
    """
    Returns the standard deviation of opinion in the nework
    """
    n = np.shape(network)
    return np.std(opinion_list(network)) / np.sqrt(n[0] * n[1])


def opinion_share(network):
    """
    Returns the share of opinion of the entire network.
    """
    unique_elements, counts = np.unique(
        opinion_list(network), return_counts=True
    )
    share = counts / np.sum(counts)
    return  pd.DataFrame([share], columns=unique_elements)



def local_clustering(voter, network):
    """
    Returns the local clustering (i.e. the fraction of agreement between one voter and its neighbors)
    """
    c = 0  # counter
    n = voter.get_number_of_neighbors()  # number of neighbours of voter
    my_opinion = voter.get_opinion()  # voter's opinion
    my_neighbors = voter.get_neighbors()  # list of neighbor coordinates
    for coordinate in my_neighbors:
        if (
            my_opinion == network[coordinate[0]][coordinate[1]].get_opinion()
        ):  # compare voters opinion with neighbor's opinion
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
    my_neighbors = voter.get_neighbors()  # list of neighbor coordinates
    n = len(my_neighbors)
    for coordinate in my_neighbors:
        s += network[coordinate[0]][
            coordinate[1]
        ].get_opinion()  # compare voters opinion with neighbor's opinion
    return s / n


def neighbor_opinion_distribution(network, name="neighbor_distribution"):
    """
    Displays the distribution of the average opinion of neighbors depending on the opinon of the voter. Returns the the mean and standard deviation for each catagory.
    """
    n = np.shape(network)
    neigh_opinion_dist = {-1: [], 0: [], 1: []}
    neigh_opinion_values = {-1: [], 0: [], 1: []}
    for i in range(n[0]):
        for j in range(n[1]):
            voter = network[i][j]
            opinion = voter.get_opinion()
            neigh_opinion = neighbor_opinion(voter, network)
            neigh_opinion_dist[opinion].append(neigh_opinion)
    plt.figure()
    for i in neigh_opinion_dist:
        avg = np.average(neigh_opinion_dist[i])
        std = np.std(neigh_opinion_dist[i])
        neigh_opinion_values[i].append([avg, std])
        plt.hist(
            neigh_opinion_dist[i],
            alpha=0.5,
            label=f"Opinion {i}: ave = {avg:.2f}, std = {std:.2f}",
        )
    plt.legend()
    plt.xlabel("xi")
    plt.ylabel("Frequency of xi")
    plt.savefig("figures/" + name + ".pdf")
    return neigh_opinion_values


def power_law(x, a, b):
    """
    Fit function for degree distribution. For y scale free network 2 < b < 3.
    """
    return a * np.power(x, b)


def deg_distribution(network):
    """
    Fit the degree distribution of the network to a power law and plot the results. Returns the exponent of the power law fit.
    """
    k = np.array([])  # list of the degrees of each node
    n = np.shape(network)
    for i in range(n[0]):
        for j in range(n[1]):
            k = np.append(
                k, network[i][j].get_number_of_neighbors()
            )  # add the nuber of neighbors for each node to k
    unique_elements, counts = np.unique(
        k, return_counts=True
    )  # count unique elements in k
    params, covariance = curve_fit(
        power_law, unique_elements, counts
    )  # fit degree distribution to power law
    a, b = params
    # plot the results
    xvals = np.linspace(min(unique_elements), max(unique_elements), num=100)
    yvals = power_law(xvals, a, b)
    plt.figure()
    plt.plot(xvals, yvals, "k--", label=f"exponent b = {b}")
    plt.plot(unique_elements, counts, "ko", label="degree distribution")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.legend()
    plt.savefig("figures/deg_distribution.pdf")
    return b


def number_media_distribution(network):
    """
    Displays the distribution of media connections of each voter. Returns the average and standard deviation.
    """
    n = np.shape(network)
    m_conx = []
    for i in range(n[0]):
        for j in range(n[1]):
            voter = network[i][j]
            m_conx.append(len(voter.get_media_connections()))
    avg = np.average(m_conx)
    std = np.std(m_conx)
    plt.figure()
    plt.hist(
        m_conx, alpha=0.5, label=f"Media connections: ave = {avg:.2f}, std = {std:.2f}"
    )
    plt.legend()
    plt.xlabel("nm")
    plt.ylabel("Frequency of nm")
    plt.savefig("figures/number_media_distribution.pdf")
    return avg, std


def opinion_trend(op_trend):
    """
    plots the opinion share
    """
    plt.figure()
    for column in op_trend.columns:
        plt.plot(op_trend.index, op_trend[column], label=f"Opinion {column}")
    plt.xlabel("Days")
    plt.ylabel("Opinion Share")
    plt.legend()
    plt.savefig("figures/opinion_share.pdf")


def plot_polarizaiton(network_pol):
    """
    plots the network polarization
    """
    plt.figure()
    plt.plot(network_pol, label="Network Polarization")
    plt.xlabel("Days")
    plt.ylabel("Polarization")
    plt.legend()
    plt.savefig("figures/network_polarization.pdf")

def plot_std(network_std):
    """
    plots the standard deviation of the network polarizaiton
    """
    plt.figure()
    plt.plot(network_std, label="Standard deviation of network polarization")
    plt.xlabel("Days")
    plt.ylabel("Std")
    plt.legend()
    plt.savefig("figures/network_std.pdf")

def plot_prob_to_change(prob_to_change):
    """
    Plots the probability to change the opinions
    """
    plt.figure()
    plt.plot(prob_to_change, label="Probability to change the opinon")
    plt.xlabel("Years")
    plt.ylabel("P")
    plt.legend()
    plt.savefig("figures/prob_to_change.pdf")
