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
    expected_columns = [-1, 0, 1]  # Replace with your expected unique elements
    unique_elements, counts = np.unique(
        opinion_list(network), return_counts=True)
    share = counts / np.sum(counts)

    # Create a DataFrame with the calculated shares
    df = pd.DataFrame([share], columns=unique_elements)

    # Ensure all expected columns are present, filling missing ones with 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder the columns to match the expected order
    df = df[expected_columns]

    return df


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
    std_opinion = np.std([m.get_opinion() for m in media])/np.sqrt(len(media))
    unique_elements, counts = np.unique(
        [m.get_category() for m in media], return_counts=True
    )

    shares = dict(zip(unique_elements, counts / np.sum(counts)))

    # Ensure 'blue', 'neutral', 'red' are in the dictionary with a default value of 0 if not found
    blue_share = shares.get("blue", 0)
    neutral_share = shares.get("neutral", 0)
    red_share = shares.get("red", 0)

    return pd.DataFrame(
        {"mean": [mean_opinion], "std": [std_opinion], "blue": [
            blue_share], "neutral": [neutral_share], "red": [red_share]}
    )


def plot_media_shares(df_stats, output_folder, file_name_shares="media_statistics_shares.pdf"):
    """
    Plot the time series of media shares for 'blue', 'neutral', and 'red'.

    This function generates a line plot of the shares over time for the three categories 
    ('blue', 'neutral', 'red') and saves the resulting figure as a PDF file.

    Parameters
    ----------
    df_stats : pandas.DataFrame
        A DataFrame indexed by time (or days) containing the columns:
        - 'blue': Share values for the "Blue" category.
        - 'neutral': Share values for the "Neutral" category.
        - 'red': Share values for the "Red" category.
    output_folder : str
        Path to the directory where the output PDF file will be saved.
    file_name_shares : str, optional
        Name of the PDF file to save the plot, by default "media_statistics_shares.pdf".

    Returns
    -------
    None
        Saves the plot as a PDF file in the specified folder.
    """

    os.makedirs(output_folder, exist_ok=True)
    output_path_shares = os.path.join(output_folder, file_name_shares)

    x_values = df_stats.index  # Days from DataFrame index

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, df_stats["blue"], label="Blue", color="blue", alpha=0.8)
    ax.plot(x_values, df_stats["neutral"],
            label="Neutral", color="gray", alpha=0.8)
    ax.plot(x_values, df_stats["red"], label="Red", color="red", alpha=0.8)
    ax.set_title("Shares Over Time")
    ax.set_xlabel("Days")
    ax.set_ylabel("Shares")
    ax.legend()
    ax.set_ylim(0, max(df_stats[["blue", "neutral", "red"]].max()) * 1.1)
    plt.tight_layout()
    plt.savefig(output_path_shares)


def plot_media_stats(df_stats, output_folder, file_name_mean="media_statistics_mean.pdf"):
    """
    Plot the time series of the mean and standard deviation for media statistics.

    This function generates a line plot for the mean of the data over time and overlays 
    a shaded region representing the mean ± standard deviation. The resulting figure 
    is saved as a PDF file.

    Parameters
    ----------
    df_stats : pandas.DataFrame
        A DataFrame indexed by time (or days) containing the columns:
        - 'mean': Mean values of the media statistics.
        - 'std': Standard deviation of the media statistics.
    output_folder : str
        Path to the directory where the output PDF file will be saved.
    file_name_mean : str, optional
        Name of the PDF file to save the plot, by default "media_statistics_mean.pdf".

    Returns
    -------
    None
        Saves the plot as a PDF file in the specified folder.

    Notes
    -----
    - The shaded region represents the range of `mean ± std`.
    - The x-axis corresponds to the index of `df_stats`, which is expected to represent time (e.g., days).
    """

    os.makedirs(output_folder, exist_ok=True)
    output_path_mean = os.path.join(output_folder, file_name_mean)

    x_values = df_stats.index  # Days from DataFrame index

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_values, df_stats["mean"], label="Mean",
            color="black", linestyle="--")
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
    # ax.set_ylim(min(df_stats["mean"]+df_stats["std"]), max((df_stats["mean"] + df_stats["std"]).max()) * 1.1)
    plt.tight_layout()
    plt.savefig(output_path_mean)


def print_media_statistics(df_stats, output_folder):
    """Exports the media statistics"""
    os.makedirs(output_folder, exist_ok=True)
    df_stats.to_csv(output_folder + "/media_statistics.csv", index=False)


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
    plt.xlabel(r"$t [\mathrm{d}]$")

    # Label the y-axis (opinion share)
    plt.ylabel("Opinion Share")

    # Display the legend
    plt.legend()

    # Save the plot to the specified file
    plt.savefig(output_path)


def voter_trend(op_trend, output_folder, file_name):
    """
    Plot the opinion share of -1 and 1 over time.

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

    df_voters = op_trend[[-1, 1]]
    df_voters = df_voters.div(df_voters.sum(axis=1), axis=0)

    # Iterate over the columns (opinions) in the DataFrame
    for column in df_voters.columns:
        # Assign colors based on the opinion value
        if column == -1:
            color = 'blue'
        elif column == 1:
            color = 'red'

        # Plot the opinion share over time
        plt.plot(df_voters.index, df_voters[column],
                 label=f"Opinion {column}", color=color)

    # Label the x-axis (time)
    plt.xlabel(r"$t [\mathrm{d}]$")

    # Label the y-axis (opinion share)
    plt.ylabel("Voter Share")

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
    plt.xlabel(r"$t [\mathrm{d}]$")

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
    plt.xlabel(r"$t [\mathrm{d}]$")

    # Label the y-axis (standard deviation)
    plt.ylabel(r"$\sigma$")

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
        plt.plot(
            prob_to_change[:, 0],
            prob_to_change[:, 1],
            label="Probability to change the opinon",
            color="black",
        )
        plt.xlabel(r"$t [\mathrm{d}]$")

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
    plt.xlabel(r"$t [\mathrm{d}]$")

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


def print_election_results(election_results, folder, filename):
    """
    Saves a list to a specified file.

    Args:
        election_results (list): The list to save.
        folder (str): The directory where the file will be saved.
        filename (str): The name of the file.
    """
    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(folder, filename)

    # Write the list to the file
    if len(election_results) > 0:
        with open(file_path, "w") as file:
            for item in election_results:
                file.write(f"{item}\n")
    else:
        with open(file_path, "w") as file:
            file.write(f"no elections were held in this simulation.")


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


def plot_consecutive_terms_histogram(df, output_folder, file_name):
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

    df['group_index'] = df.index // 28

    # Group by the new index and sum the values
    df_grouped = df.groupby('group_index').sum()

    df = df_grouped

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_name)

    if not {1, -1}.issubset(df.columns):
        raise ValueError("The DataFrame must have columns named 1 and -1.")

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(df.index + 1.2, df[1], width=0.4,
            color="red", label="1", align="center", alpha=0.8)
    plt.bar(df.index + 0.8, df[-1], width=0.4,
            color="blue", label="-1", align="center", alpha=0.8)

    # Add titles and labels
    plt.xticks(ticks=range(int(df.index.min())+1, int(df.index.max()) + 2))
    plt.title("Histogram of consecutive terms", fontsize=14)
    plt.xlabel("Number of consecutive terms", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(loc="best")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path)
