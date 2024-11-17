import numpy as np
import pandas as pd
import os
import datetime
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


def make_foldername(base_name="figures"):
    """
    Creates a folder name based on the current date (year, month, day).
    Appends a sequential number if folders with the same base name already exist.

    Parameters:
    - base_name (str): The base name for the folder.

    Returns:
    - str: A unique folder name.
    """
    # Get current date in YYYYMMDD format
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    base_foldername = f"{base_name}_{date_str}"

    # Check for existing folders with the same base name
    counter = 1
    foldername = base_foldername
    while os.path.exists(foldername):
        foldername = f"{base_foldername}_{counter}"
        counter += 1
    
    return foldername


def neighbor_opinion_distribution(network, output_folder, file_name):
    """
    Displays the distribution of the average opinion of neighbors depending on the opinon of the voter. Returns the the mean and standard deviation for each catagory.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
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
        if i == -1:
            c = 'blue'
        elif i == 0:
            c = 'grey'
        else:
            c = 'red'
        avg = np.average(neigh_opinion_dist[i])
        std = np.std(neigh_opinion_dist[i])
        neigh_opinion_values[i].append([avg, std])
        plt.hist(
            neigh_opinion_dist[i],
            alpha=0.5,
            label=f"Opinion {i}: ave = {avg:.2f}, std = {std:.2f}",
            color=c
        )
    plt.legend()
    plt.xlabel("$x_i^N$")
    plt.ylabel("$f(x_i^N)$")
    plt.savefig(output_path)
    return neigh_opinion_values


def power_law(x, a, b):
    """
    Fit function for degree distribution. For y scale free network 2 < b < 3.
    """
    return a * np.power(x, b)


def deg_distribution(network, output_folder, file_name):
    """
    Fit the degree distribution of the network to a power law and plot the results. Returns the exponent of the power law fit.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
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
    plt.plot(xvals, yvals, "k--", label=f"exponent $b = {b}$")
    plt.plot(unique_elements, counts, "ko", label="degree distribution")
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    plt.legend()
    plt.savefig(output_path)
    return b


def number_media_distribution(network, output_folder, file_name):
    """
    Displays the distribution of media connections of each voter. Returns the average and standard deviation.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
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
    plt.xlabel("$n_m$")
    plt.ylabel("$f(n_m)$")
    plt.savefig(output_path)
    return avg, std


def opinion_trend(op_trend, output_folder, file_name):
    """
    plots the opinion share
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
    plt.figure()
    for column in op_trend.columns:
        if column == -1:
            c = 'blue'
        elif column == 0:
            c = 'grey'
        else:
            c = 'red'
        plt.plot(op_trend.index, op_trend[column], label=f"Opinion {column}", color=c)
    plt.xlabel("$t [\mathrm{d}]$")
    plt.ylabel("Opinion Share")
    plt.legend()
    plt.savefig(output_path)


def plot_polarizaiton(network_pol, output_folder, file_name):
    """
    plots the network polarization
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
    plt.figure()
    plt.plot(network_pol, label="Network Polarization", color='black')
    plt.xlabel("$t [\mathrm{d}]$")
    plt.ylabel("$S$")
    plt.legend()
    plt.savefig(output_path)

def plot_std(network_std, output_folder, file_name):
    """
    plots the standard deviation of the network polarizaiton
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
    plt.figure()
    plt.plot(network_std, label="Standard deviation of network polarization", color='black')
    plt.xlabel("$t [\mathrm{d}]$")
    plt.ylabel("$\sigma$")
    plt.legend()
    plt.savefig(output_path)

def plot_prob_to_change(prob_to_change, output_folder, file_name):
    """
    Plots the probability to change the opinions
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
    if len(prob_to_change) > 1:
        plt.figure()
        prob_to_change = np.array(prob_to_change)
        plt.plot(prob_to_change[:,0], prob_to_change[:,1], label="Probability to change the opinon", color='black')
        plt.xlabel("$t [\mathrm{d}]$")
        plt.ylabel("$P$")
        plt.legend()
        plt.savefig(output_path)

def plot_clustering(clustering, output_folder, file_name):
    """
    Plots the clustering
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
    plt.figure()
    plt.plot(clustering, label="Network clustering", color='black')
    plt.xlabel("$t [\mathrm{d}]$")
    plt.ylabel("$C_v$")
    plt.legend()
    plt.savefig(output_path)

def print_parameters(args, output_folder, file_name):
    """
    Saves all arguments from argparse to a text file.

    Parameters:
    - args: Namespace object from argparse.parse_args().
    - file_name: The name of the text file to save arguments.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
    with open(output_path, 'w') as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")
