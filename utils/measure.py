"""
This module contains lots of usefule functions to gain information about the network.
"""

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
    unique_elements, counts = np.unique(opinion_list(network), return_counts=True)
    share = counts / np.sum(counts)
    return pd.DataFrame([share], columns=unique_elements)


def media_statistics(media):
    """
    Returns a dataframe with the statistics of the current media landscape

    Parameter
    ---------
    media: list
        A list of Media objects

    Returns
    -------
    pd.DataFrame
        With columns ["mean", "std", "blue", "red", "neutral"]
        which contain the mean, the standard deviation, and the share of "blue", "neutral" and "red" media nodes.
    """
    mean_opinion = np.mean([m.get_opinion() for m in media])
    std_opinion = np.std([m.get_opinion() for m in media])
    unique_elements, counts = np.unique(
        [m.get_category() for m in media], return_counts=True
    )
    shares = dict(zip(unique_elements, counts / np.sum(counts)))


    return pd.DataFrame(
        {"mean":[mean_opinion],"std": [std_opinion],"blue": [shares["blue"]],"neutral": [shares["neutral"]],"red": [shares["red"]]}
    )


def plot_media_statistics(df_stats, output_folder, file_name_shares="media_statistics_shares.pdf", file_name_mean="media_statistics_mean.pdf"):
    """
    Plots a summary DataFrame over time (index as x-axis) with 'mean', 'std', and shares for 'blue', 'neutral', and 'red'.
    Uses lines for shares and a shaded region for 'mean ± std'.
    
    Parameters:
        df (pd.DataFrame): A DataFrame indexed by time (or days) with 'mean', 'std', 'blue', 'neutral', and 'red'.
    """

    os.makedirs(output_folder, exist_ok=True)
    output_path_shares = os.path.join(output_folder, file_name_shares)
    output_path_mean = os.path.join(output_folder, file_name_mean)


    x_values = df_stats.index  # Days from DataFrame index

    # Plot 1: Shares
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, df_stats["blue"], label="Blue", color="blue", alpha = 0.8)
    ax.plot(x_values, df_stats["neutral"], label="Neutral", color="gray", alpha = 0.8)
    ax.plot(x_values, df_stats["red"], label="Red", color="red", alpha = 0.8)
    ax.set_title("Shares Over Time")
    ax.set_xlabel("Days")
    ax.set_ylabel("Shares")
    ax.legend()
    ax.set_ylim(0, max(df_stats[["blue", "neutral", "red"]].max()) * 1.1)
    plt.tight_layout()
    plt.savefig(output_path_shares)
    plt.show()

    # Plot 2: Mean ± Std
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, df_stats["mean"], label="Mean", color="black", linestyle="--")
    ax.fill_between(
        x_values,
        df_stats["mean"] - df_stats["std"],
        df_stats["mean"] + df_stats["std"],
        color="lightgreen",
        alpha=0.2,
        label="Mean ± Std"
    )
    ax.set_title("Mean ± Std Over Time")
    ax.set_xlabel("Days")
    ax.set_ylabel("Values")
    ax.legend()
    ax.set_ylim(min(df_stats["mean"]+df_stats["std"]), max((df_stats["mean"] + df_stats["std"]).max()) * 1.1)
    plt.tight_layout()
    plt.savefig(output_path_mean)
    plt.show()
    
def print_media_statistics(df_stats, output_folder):
    """Exports the media statistics"""
    os.makedirs(output_folder, exist_ok=True)
    df_stats.to_csv(output_folder + "/media_statistics.csv", index=False)


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
            c = "blue"
        elif i == 0:
            c = "grey"
        else:
            c = "red"
        avg = np.average(neigh_opinion_dist[i])
        std = np.std(neigh_opinion_dist[i])
        neigh_opinion_values[i].append([avg, std])
        plt.hist(
            neigh_opinion_dist[i],
            alpha=0.5,
            label=f"Opinion {i}: ave = {avg:.2f}, std = {std:.2f}",
            color=c,
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
            c = "blue"
        elif column == 0:
            c = "grey"
        else:
            c = "red"
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
    plt.plot(network_pol, label="Network Polarization", color="black")
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
    plt.plot(
        network_std, label="Standard deviation of network polarization", color="black"
    )
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
        plt.plot(
            prob_to_change[:, 0],
            prob_to_change[:, 1],
            label="Probability to change the opinon",
            color="black",
        )
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
    plt.plot(clustering, label="Network clustering", color="black")
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
    with open(output_path, "w") as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")


def print_measure(measure, output_folder, file_name):
    """
    prints measure to file
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
    with open(output_path, "w") as file:
        for m in measure:
            file.write(f"{m} \n")


def print_prob_to_change(prob_to_change, output_folder, file_name):
    """
    prints to file
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)
    with open(output_path, "w") as file:
        for m in prob_to_change:
            file.write(f"{m[0]} \t {m[1]} \n")


def get_consecutive_terms_counts(election_results):
    """
    Generate a DataFrame with counts of consecutive occurrences of 1 and -1.

    Parameters:
        election_results (list): A list containing 1s and -1s representing the terms each party won.

    Returns:
        DataFrame: A DataFrame with 'Consecutive Elections' as the index (from 0 to the maximum run length found),
                   and columns for the counts of consecutive occurrences of 1 and -1.
    """
    if not election_results:
        # If no data, return an empty DataFrame with index starting from 0
        return pd.DataFrame(columns=[-1, 1], index=pd.RangeIndex(start=0, stop=1))

    # Initialize a list to store the counts of consecutive occurrences
    counts = []
    current_value = election_results[0]
    count = 1

    # Loop through the data to calculate streak lengths
    for value in election_results[1:]:
        if value == current_value:
            count += 1
        else:
            counts.append((current_value, count))
            current_value = value
            count = 1
    counts.append((current_value, count))  # Add the last streak

    # Create a dictionary to store counts for consecutive occurrences
    consecutive_dict = {-1: {}, 1: {}}
    for value, count in counts:
        if count not in consecutive_dict[value]:
            consecutive_dict[value][count] = 0
        consecutive_dict[value][count] += 1

    # Find the maximum run length
    max_run_length = max(
        max(consecutive_dict[1].keys(), default=0),
        max(consecutive_dict[-1].keys(), default=0),
    )

    # Create a DataFrame with index from 0 to the maximum run length
    df = pd.DataFrame(index=range(1, max_run_length + 1))
    df.index.name = "Consecutive Elections"

    # Fill the columns for 1 and -1 with counts or 0 if not present
    df[1] = [consecutive_dict[1].get(i, 0) for i in df.index]
    df[-1] = [consecutive_dict[-1].get(i, 0) for i in df.index]

    return df


def create_consecutive_terms_histogram(df):
    """
    Creates a histogram from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame with two columns, `1` and `-1`, and integer indices.
        The index represents the bins.

    Returns
    -------
    None
        This function does not return any value. It displays a histogram
        where each column is plotted in a different color (red for `1` and blue for `-1`).
    """

    if not {1, -1}.issubset(df.columns):
        raise ValueError("The DataFrame must have columns named 1 and -1.")

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, df[1], width=0.4, color="red", label="1", align="center")
    plt.bar(df.index - 0.4, df[-1], width=0.4, color="blue", label="-1", align="center")

    # Add titles and labels
    plt.title("Histogram of Consecutive Terms", fontsize=14)
    plt.xlabel("Number of consecutive terms", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(loc="best")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
