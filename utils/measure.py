import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def opinion_list(network):
    """
    Returns a list of all opinions in the network.

    Parameters
    ----------
    network : list of list of Voter
        The voter network where each element is a `Voter` object.

    Returns
    -------
    list of int
        A flat list of opinions from all voters in the network.
    """
    opinions = []
    n = np.shape(network)
    for i in range(n[0]):
        for j in range(n[1]):
            opinions.append(network[i][j].get_opinion())
    return opinions


def polarization(network):
    """
    Calculates the average opinion of the network.

    Parameters
    ----------
    network : list of list of Voter
        The voter network.

    Returns
    -------
    float
        The average opinion of all voters in the network.
    """
    return np.mean(opinion_list(network))


def std_opinion(network):
    """
    Computes the standard deviation of opinions in the network.

    Parameters
    ----------
    network : list of list of Voter
        The voter network.

    Returns
    -------
    float
        The standard deviation of opinions, normalized by the square root of the network size.
    """
    n = np.shape(network)
    return np.std(opinion_list(network)) / np.sqrt(n[0] * n[1])


def opinion_share(network):
    """
    Calculates the share of each opinion in the network.

    Parameters
    ----------
    network : list of list of Voter
        The voter network.

    Returns
    -------
    pd.DataFrame
        A DataFrame where columns are unique opinions and values represent their share.
    """
    unique_elements, counts = np.unique(
        opinion_list(network), return_counts=True)
    share = counts / np.sum(counts)
    return pd.DataFrame([share], columns=unique_elements)


def local_clustering(voter, network):
    """
    Computes the local clustering coefficient for a voter.

    Parameters
    ----------
    voter : Voter
        The voter for whom local clustering is calculated.
    network : list of list of Voter
        The voter network.

    Returns
    -------
    float
        The fraction of neighbors with the same opinion as the voter.
    """
    c = 0  # Agreement counter
    n = voter.get_number_of_neighbors()
    my_opinion = voter.get_opinion()
    my_neighbors = voter.get_neighbors()

    for coordinate in my_neighbors:
        if my_opinion == network[coordinate[0]][coordinate[1]].get_opinion():
            c += 1
    return c / n


def clustering(network):
    """
    Computes the average clustering coefficient for the entire network.

    Parameters
    ----------
    network : list of list of Voter
        The voter network.

    Returns
    -------
    float
        The average clustering coefficient across all voters in the network.
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
    Calculates the average opinion of a voter's neighbors.

    Parameters
    ----------
    voter : Voter
        The voter whose neighbors are being analyzed.
    network : list of list of Voter
        The voter network.

    Returns
    -------
    float
        The average opinion of the voter's neighbors.
    """
    s = 0
    my_neighbors = voter.get_neighbors()
    n = len(my_neighbors)

    for coordinate in my_neighbors:
        s += network[coordinate[0]][coordinate[1]].get_opinion()
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
    Plots the distribution of the average opinion of neighbors based on voter opinions.

    Parameters
    ----------
    network : list of list of Voter
        The voter network.
    output_folder : str
        Path to the folder where the plot will be saved.
    file_name : str
        Name of the output file.

    Returns
    -------
    dict
        A dictionary with opinions as keys and lists of average opinions and standard deviations as values.
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
        c = 'blue' if i == -1 else 'grey' if i == 0 else 'red'
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
    Fit function for degree distribution in a network.

    Parameters
    ----------
    x : array_like
        Input data points (degrees).
    a : float
        Scale parameter of the power law.
    b : float
        Exponent of the power law.

    Returns
    -------
    array_like
        Power-law values for the input data.
    """
    return a * np.power(x, b)


def deg_distribution(network, output_folder, file_name):
    """
    Fits the degree distribution of the network to a power law and plots the results.

    Parameters
    ----------
    network : list of list of Voter
        The voter network where each Voter has a defined number of neighbors.
    output_folder : str
        Path to the folder where the plot will be saved.
    file_name : str
        Name of the output file.

    Returns
    -------
    float
        The exponent of the power law fit.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)

    # Collect the degrees of all nodes in the network
    k = np.array([])  # Stores the degree of each node
    n = np.shape(network)  # Get the dimensions of the network
    for i in range(n[0]):
        for j in range(n[1]):
            # Append the number of neighbors for each node
            k = np.append(k, network[i][j].get_number_of_neighbors())

    # Compute the unique degrees and their frequencies
    unique_elements, counts = np.unique(k, return_counts=True)

    # Fit the degree distribution to a power-law function
    params, _ = curve_fit(power_law, unique_elements, counts)
    a, b = params  # Extract the scale and exponent parameters

    # Plot the degree distribution and fitted power-law curve
    xvals = np.linspace(min(unique_elements), max(unique_elements), num=100)
    yvals = power_law(xvals, a, b)
    plt.figure()
    # Fitted curve
    plt.plot(xvals, yvals, "k--", label=f"Exponent $b = {b:.2f}$")
    plt.plot(unique_elements, counts, "ko",
             label="Degree distribution")  # Data points
    plt.xlabel("$k$")  # Degree of nodes
    plt.ylabel("$P(k)$")  # Probability of nodes with degree k
    plt.legend()
    plt.savefig(output_path)  # Save the plot
    return b  # Return the power-law exponent


def number_media_distribution(network, output_folder, file_name):
    """
    Displays the distribution of media connections of voters.

    Parameters
    ----------
    network : list of list of Voter
        The voter network where each Voter has media connections.
    output_folder : str
        Path to the folder where the plot will be saved.
    file_name : str
        Name of the output file.

    Returns
    -------
    tuple
        The average and standard deviation of media connections.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)

    # Collect the number of media connections for all voters in the network
    m_conx = [len(voter.get_media_connections())
              for row in network for voter in row]

    # Calculate average and standard deviation of media connections
    avg = np.mean(m_conx)
    std = np.std(m_conx)

    # Plot the histogram of media connections
    plt.figure()
    plt.hist(
        m_conx,
        alpha=0.5,
        label=f"Media connections: ave = {avg:.2f}, std = {std:.2f}"
    )
    plt.legend()  # Add a legend with summary statistics
    plt.xlabel("$n_m$")  # Media connections of each voter
    plt.ylabel("$f(n_m)$")  # Frequency of media connections
    plt.savefig(output_path)  # Save the plot

    return avg, std  # Return the average and standard deviation of media connections


def opinion_trend(op_trend, output_folder, file_name):
    """
    Plot the opinion share over time.

    Parameters
    ----------
    op_trend : pandas.DataFrame
        DataFrame where columns represent different opinions and the index is time.
    output_folder : str
        Path to the folder where the plot will be saved.
    file_name : str
        Name of the output file where the plot will be saved.

    Notes
    -----
    This function assigns specific colors to the opinions: blue for -1, grey for 0,
    and red for other opinions. It then plots the opinion share over time and saves the plot.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full output file path
    output_path = os.path.join(output_folder, file_name)

    # Create a new figure for the plot
    plt.figure()

    # Iterate over the columns (opinions) in the DataFrame
    for column in op_trend.columns:
        # Assign colors based on the opinion value
        if column == -1:
            color = 'blue'
        elif column == 0:
            color = 'grey'
        else:
            color = 'red'

        # Plot the opinion share over time
        plt.plot(op_trend.index, op_trend[column],
                 label=f"Opinion {column}", color=color)

    # Label the x-axis (time)
    plt.xlabel("$t [\mathrm{d}]$")

    # Label the y-axis (opinion share)
    plt.ylabel("Opinion Share")

    # Display the legend
    plt.legend()

    # Save the plot to the specified file
    plt.savefig(output_path)


def plot_polarization(network_pol, output_folder, file_name):
    """
    Plot the network polarization over time.

    Parameters
    ----------
    network_pol : array-like
        Array or list of network polarization values over time.
    output_folder : str
        Path to the folder where the plot will be saved.
    file_name : str
        Name of the output file where the plot will be saved.

    Notes
    -----
    This function plots the network polarization over time and saves the plot.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full output file path
    output_path = os.path.join(output_folder, file_name)

    # Create a new figure for the plot
    plt.figure()

    # Plot the network polarization
    plt.plot(network_pol, label="Network Polarization", color='black')

    # Label the x-axis (time)
    plt.xlabel("$t [\mathrm{d}]$")

    # Label the y-axis (polarization)
    plt.ylabel("$S$")

    # Display the legend
    plt.legend()

    # Save the plot to the specified file
    plt.savefig(output_path)


def plot_std(network_std, output_folder, file_name):
    """
    Plot the standard deviation of the network polarization over time.

    Parameters
    ----------
    network_std : array-like
        Array or list of standard deviation values of network polarization over time.
    output_folder : str
        Path to the folder where the plot will be saved.
    file_name : str
        Name of the output file where the plot will be saved.

    Notes
    -----
    This function plots the standard deviation of the network polarization and saves the plot.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full output file path
    output_path = os.path.join(output_folder, file_name)

    # Create a new figure for the plot
    plt.figure()

    # Plot the standard deviation of network polarization
    plt.plot(
        network_std, label="Standard deviation of network polarization", color='black')

    # Label the x-axis (time)
    plt.xlabel("$t [\mathrm{d}]$")

    # Label the y-axis (standard deviation)
    plt.ylabel("$\sigma$")

    # Display the legend
    plt.legend()

    # Save the plot to the specified file
    plt.savefig(output_path)


def plot_prob_to_change(prob_to_change, output_folder, file_name):
    """
    Plot the probability to change opinions (per year) over time.

    Parameters
    ----------
    prob_to_change : list of tuples
        A list of tuples where each tuple contains time and corresponding probability to change opinion.
    output_folder : str
        Path to the folder where the plot will be saved.
    file_name : str
        Name of the output file where the plot will be saved.

    Notes
    -----
    This function plots the probability of opinion change over time and saves the plot.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full output file path
    output_path = os.path.join(output_folder, file_name)

    # Ensure there is more than one data point to plot
    if len(prob_to_change) > 1:
        # Create a new figure for the plot
        plt.figure()

        # Convert the probability data into a numpy array for easier manipulation
        prob_to_change = np.array(prob_to_change)

        # Plot the probability of opinion change over time
        plt.plot(prob_to_change[:, 0], prob_to_change[:, 1],
                 label="Probability to change the opinion", color='black')

        # Label the x-axis (time)
        plt.xlabel("$t [\mathrm{d}]$")

        # Label the y-axis (probability)
        plt.ylabel("$P$")

        # Display the legend
        plt.legend()

        # Save the plot to the specified file
        plt.savefig(output_path)


def plot_clustering(clustering, output_folder, file_name):
    """
    Plot the network clustering coefficient over time.

    Parameters
    ----------
    clustering : array-like
        Array or list of clustering coefficient values over time.
    output_folder : str
        Path to the folder where the plot will be saved.
    file_name : str
        Name of the output file where the plot will be saved.

    Notes
    -----
    This function plots the network clustering coefficient over time and saves the plot.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full output file path
    output_path = os.path.join(output_folder, file_name)

    # Create a new figure for the plot
    plt.figure()

    # Plot the network clustering coefficient
    plt.plot(clustering, label="Network clustering", color='black')

    # Label the x-axis (time)
    plt.xlabel("$t [\mathrm{d}]$")

    # Label the y-axis (clustering coefficient)
    plt.ylabel("$C_v$")

    # Display the legend
    plt.legend()

    # Save the plot to the specified file
    plt.savefig(output_path)


def print_parameters(args, output_folder, file_name):
    """
    Save all command line arguments to a text file.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace object containing the parsed arguments.
    output_folder : str
        Path to the folder where the file will be saved.
    file_name : str
        Name of the output text file.

    Notes
    -----
    This function saves all arguments parsed from `argparse.parse_args()` to a text file.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full output file path
    output_path = os.path.join(output_folder, file_name)

    # Open the output file and write the arguments to it
    with open(output_path, 'w') as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")


def print_measure(measure, output_folder, file_name):
    """
    Print a list of measures to a file.

    Parameters
    ----------
    measure : list
        List of measurements to be printed.
    output_folder : str
        Path to the folder where the file will be saved.
    file_name : str
        Name of the output text file.

    Notes
    -----
    This function prints each item from the `measure` list to a text file.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full output file path
    output_path = os.path.join(output_folder, file_name)

    # Open the output file and write the measures to it
    with open(output_path, 'w') as file:
        for m in measure:
            file.write(f"{m} \n")


def print_prob_to_change(prob_to_change, output_folder, file_name):
    """
    Print the probability to change opinions over time to a file.

    Parameters
    ----------
    prob_to_change : list of tuples
        A list of tuples where each tuple contains time and corresponding probability to change opinion.
    output_folder : str
        Path to the folder where the file will be saved.
    file_name : str
        Name of the output text file.

    Notes
    -----
    This function prints the probability of opinion change over time to a text file.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full output file path
    output_path = os.path.join(output_folder, file_name)

    # Open the output file and write the data to it
    with open(output_path, 'w') as file:
        for m in prob_to_change:
            file.write(f"{m[0]} \t {m[1]} \n")
