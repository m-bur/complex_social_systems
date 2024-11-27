import random
import numpy as np
import pandas as pd
import random
from utils.measure import polarization
from utils.measure import opinion_share

from utils.nodes import *


def init_df_conx(c_min, c_max, gamma, L):
    """
    Initialize the connection dataframe.

    Creates a dataframe with `L * L` rows representing nodes. The dataframe has the following columns:
    - `x`: x-coordinate of the node.
    - `y`: y-coordinate of the node.
    - `number of connections`: The total number of connections for the node.
    - `numbers left (auxiliary)`: An auxiliary column for remaining connections.
    - `connection list`: A list of connections for the node.

    Parameters
    ----------
    c_min : int
        Minimal number of connections.
    c_max : int
        Maximal number of connections.
    gamma : float
        Distribution parameter for the number of connections (higher gamma results in fewer connections).
    L : int
        Side length of the grid; total number of nodes is `N = L * L`.

    Returns
    -------
    pandas.DataFrame
        A dataframe representing the connections.
    """

    # Get the probability list between c_min and c_max
    prob_list = []
    for k in range(c_min, c_max + 1):
        prob = k ** (-gamma)
        prob_list.append(prob)
    prob_sum = sum(prob_list)
    prob_list_norm = [_ / prob_sum for _ in prob_list] + \
        [1]  # normalized prob list
    prob_list = [
        sum(prob_list_norm[:_]) for _ in range(len(prob_list_norm))
    ]  # accumulative prob list

    # Randomly assign number of connections for the population
    df_conx = pd.DataFrame(columns=["x", "y", "num", "num_left", "connection"])
    for i in range(L):
        for j in range(L):
            # Store the row with 'x', 'y', 'number connections', and 'connection list'
            val = random.random()
            index = (
                next(
                    _
                    for _ in range(len(prob_list))
                    if prob_list[_] <= val < prob_list[_ + 1]
                )
                + c_min
            )
            df_conx.loc[len(df_conx)] = [i, j, index, index, []]
    return df_conx


def prob_first_connection(dist, L, L_G):
    """
    Probability function to determine the existence of a first-level connection between two nodes.

    Parameters
    ----------
    dist : float
        Cartesian distance between two nodes.
    L : int
        Side length of the grid; total number of nodes is `N = L * L`.
    L_G : int
        Side length of a local group.

    Returns
    -------
    float
        Probability of a link (before normalization).
    """

    a = L_G
    b = a / 4
    prob = 1 / (1 + np.exp((dist - a) / b)) + 0.001 * (L - dist) / L
    if prob < 0:
        prob = 0
    return prob


def calc_first_conx_matrix(L, L_G):
    """
    Generate the first-level connection matrix.

    Parameters
    ----------
    L : int
        Side length of the grid; total number of nodes is `N = L * L`.
    L_G : int
        Side length of a local group.

    Returns
    -------
    numpy.ndarray
        The probability matrix of the first-level connection.
    """

    print("==== Generate first level connection")
    pop = L * L
    first_conx_matrix = np.zeros((pop, pop))
    for p in range(pop):
        i = p // L
        j = p % L
        for q in range(pop):
            if q == p:
                first_conx_matrix[p, q] = 0
            elif q > p:
                m = q // L
                n = q % L
                dist = np.sqrt((i - m) ** 2 + (j - n) ** 2)
                first_conx_matrix[p, q] = prob_first_connection(dist, L, L_G)
                first_conx_matrix[q, p] = first_conx_matrix[p, q]
    return first_conx_matrix


def calc_second_conx_matrix(L, first_conx_matrix, p_c):
    """
    Generate the second-level connection matrix.

    Parameters
    ----------
    L : int
        Side length of the grid; total number of nodes is `N = L * L`.
    first_conx_matrix : numpy.ndarray
        The first-level connection matrix.
    p_c : float
        Probability parameter for establishing a second-level connection.

    Returns
    -------
    numpy.ndarray
        The probability matrix of the second-level connection.
    """

    print("==== Generate second level connection")
    pop = L * L
    second_conx_matrix = np.zeros((pop, pop))
    for p in range(pop):
        print(p)
        for q in range(pop):
            if q == p:
                second_conx_matrix[p, q] = 0
            elif q > p:
                prob = 0
                for r in range(pop):
                    prob += first_conx_matrix[p, r] * first_conx_matrix[q, r]
                prob *= p_c
                second_conx_matrix[p, q] = prob
                second_conx_matrix[q, p] = prob
    return second_conx_matrix


def update_df_conx(L, df_conx, connection_matrix):
    """
    Generate the connection dataframe (i.e., the network).

    Parameters
    ----------
    L : int
        Side length of the grid.
    df_conx : pandas.DataFrame
        Connection dataframe.
    connection_matrix : numpy.ndarray
        The sum of the first- and second-level connection matrices.

    Returns
    -------
    pandas.DataFrame
        Updated connection dataframe representing the network.
    """

    print("==== Generate network")
    pop = L * L
    connection_matrix_norm = np.zeros((pop, pop))
    for p in range(pop):
        prob_list = connection_matrix[p, :]
        num_left = df_conx["num_left"]
        prob_list = [prob_list[_] * num_left[_] for _ in range(pop)]
        sum_prob = np.sum(prob_list)
        if sum_prob > 0:
            prob_list = [_ / sum_prob for _ in prob_list]

            i = p // L
            j = p % L
            sample = df_conx.loc[p, "num_left"]
            left_sample = len([_ for _ in prob_list if _ != 0])
            if sample > left_sample:
                sample = left_sample
            selected_idx = list(
                np.random.choice(
                    len(prob_list), size=sample, replace=False, p=prob_list
                )
            )

            for q in selected_idx:
                m = q // L
                n = q % L
                df_conx.at[p, "connection"].append((m, n))
                df_conx.at[p, "num_left"] = df_conx.at[p, "num_left"] - 1
                df_conx.at[q, "connection"].append((i, j))
                df_conx.at[q, "num_left"] = df_conx.at[q, "num_left"] - 1
                connection_matrix_norm[p, q] = 0
                connection_matrix_norm[q, p] = 0

    df_conx = df_conx[["x", "y", "num", "connection"]]
    return df_conx


def init_network(df_conx, L, media_feedback_probability, media_feedback_threshold_replacement_neutral):
    """
    Initializes a voter network based on a given connections dataframe.

    This function creates a 2D grid of voter nodes, assigns initial opinions randomly, 
    and establishes neighbors for each voter based on the connections provided in 
    `df_conx`. It returns the resulting network as a list of lists, where each 
    element represents a voter node with its initialized properties and connections.

    Parameters
    ----------
    df_conx : pandas.DataFrame
        A DataFrame containing connection information for the network. Each row 
        should include:
            - 'x': int
                X-coordinate of the voter in the grid.
            - 'y': int
                Y-coordinate of the voter in the grid.
            - 'connection': list of tuple of int
                A list of neighboring coordinates for the voter.
    media_feedback_probability : float
        Probability that a voter is influenced by media feedback.
    media_feedback_threshold_replacement_neutral : float
        Threshold for media feedback to replace a voter's opinion with a neutral one.

    Returns
    -------
    list of list of Voter
        A 2D list representing the network of voter nodes, where each voter has its 
        neighbors and initial opinion set.
    """
    # Get the size of the network from the number of rows in the DataFrame

    # Create a 2D list of Voter objects, initialized with default opinions and media feedback settings
    network = [
        [
            Voter(i, j, 0, 10, False, media_feedback_probability,
                  media_feedback_threshold_replacement_neutral)
            for i in range(L)
        ]
        for j in range(L)
    ]
    
    num_elements = L ** 2
    ones = [1] * (num_elements // 3)
    zeros = [0] * (num_elements // 3)
    minus_ones = [-1] * (num_elements // 3)
    
    # If L^2 is not perfectly divisible by 3, adjust by adding the remainder
    remainder = num_elements % 3
    if remainder == 1:
        ones.append(1)
    elif remainder == 2:
        ones.append(1)
        zeros.append(0)
    
    # Combine the three lists
    opinion_list = ones + zeros + minus_ones
    
    # Shuffle the list to randomize it
    random.shuffle(opinion_list)
    # Iterate through the DataFrame to set voter properties and establish neighbor connections
    for i in range(L**2):
        row = df_conx.iloc[i]  # Get the current row
        neighbors = row["connection"]  # List of neighboring coordinates
        # Reference the voter at the given (x, y) position
        voter = network[row["x"]][row["y"]]

        # Set an initial opinion for the voter, chosen randomly from [-1, 0, 1]
        voter.set_opinion(opinion_list[i])

        # Add each neighbor to the voter's list of neighbors
        for ncoordinates in neighbors:
            voter.add_neighbor(ncoordinates[0], ncoordinates[1])

    # Return the fully initialized network
    return network

def update_media(days, media, election_results, initial_media_opinion, number_of_days_election_cycle, x, y, media_update_cycle=1):
    """
    Updates the opinions of media entities based on daily cycles and election results.

    Parameters
    ----------
    days : int
        The current day in the simulation.
    media : list
        A list of media objects.
    election_results : list
        A list of election outcomes, where the most recent result is the last element. 
        Positive values represent red party; negative values represent the blue.
    initial_media_opinion : float
        The baseline opinion of the media.
    number_of_days_election_cycle : int
        The number of days in an election cycle.
    x : float
        A scaling factor for media change.
    y : float
        Another scaling factor for election-related media updates.
    media_update_cycle : int, optional
        The interval (in days) at which media opinions are updated. Defaults to 1.

    Returns
    -------
    list
        The updated list of media objects.

    Notes
    -----
    - Media opinion is bounded between -1 and 1.
    - Media opinions are updated daily based on a normal distribution scaled by `x`.
    - During election cycles, media opinions are further influenced by the duration
      of the ruling party's power (`DUR`) and a scaling factor (`y`).
    """
    # random (ecconomy) term
    if days % media_update_cycle == 0:
        # Generate a random opinion change using a normal distribution, scaled by `x` and baseline opinion.
        media_change = np.random.normal(initial_media_opinion, 0.00022 * x)
        
        # Update opinions for each media entity.
        for i, _ in enumerate(media):
            # Calculate the new opinion by adding the change to the current opinion.
            new_opinion = media[i].get_opinion() + media_change
            
            # Ensure the opinion stays within bounds (-1 to 1).
            if abs(new_opinion) < 1:
                media[i].set_opinion(new_opinion)
            elif new_opinion < 0:
                media[i].set_opinion(-1)
            elif new_opinion > 0:
                media[i].set_opinion(1)

    # duration term
    if days % number_of_days_election_cycle == 0:
        dur = 0  # Duration multiplier for ruling party influence.

        # Determine the duration of the ruling party's power.
        if get_number_of_consecutive_terms(election_results) >= 2:
            # Increase duration based on consecutive terms, with diminishing returns after the second term.
            dur = 1 + (get_number_of_consecutive_terms(election_results) - 2) * 0.25
            dur = min(dur, 3)  # Cap the duration multiplier at 3.
        
        # Determine the direction of influence based on the ruling party.
        if election_results:
            i = (-1) * election_results[-1] if election_results[-1] is not None else 0
        else:
            i = 0

        # Calculate the opinion change during the election cycle.
        media_change = dur * 0.000376 * x * y * i

        # Update opinions for each media entity based on election influence.
        for i, _ in enumerate(media):
            new_opinion = media[i].get_opinion() + media_change
            
            # Ensure the opinion stays within bounds (-1 to 1).
            if abs(new_opinion) < 1:
                media[i].set_opinion(new_opinion)
            elif new_opinion < 0:
                media[i].set_opinion(-1)
            elif new_opinion > 0:
                media[i].set_opinion(1)

    return media

def generate_media_landscape(
    number_of_media, mode="standard", mu=0, sigma=0.25, lower_bound=-1, upper_bound=1
):
    """
    Generate a pandas DataFrame containing media nodes and their respective IDs, 
    where the opinions of the media nodes are generated based on the specified mode.

    Parameters
    ----------
    number_of_media : int
        The number of media nodes to generate.
    mode : str, optional
        The distribution mode for generating opinions. Can be one of the following:
        - 'standard': Uniform distribution between -1 and 1 (default).
        - 'uniform': Uniform distribution between -lower_bound and upper_bound.
        - 'gaussian': Gaussian (normal) distribution with mean `mu` and standard deviation `sigma`.
        - 'fixed': equally spaced media nodes.
    mu : float, optional
        The mean of the Gaussian distribution (default is 0).
    sigma : float, optional
        The standard deviation of the Gaussian distribution (default is 0.25).
    lower_bound : float, optional
        The lower bound of the uniform distribution for 'uniform' mode (default is -1).
    upper_bound : float, optional
        The upper bound of the uniform distribution for 'uniform' mode (default is 1).

    Returns
    -------
    list
        A list that contains all the media nodes which have an opinion distribution according

    Notes
    -----
    - In 'standard' mode, opinions are uniformly distributed between -1 and 1.
    - In 'uniform' mode, opinions are uniformly distributed between -lower_bound and upper_bound.
    - In 'gaussian' mode, opinions are generated from a Gaussian distribution and clipped to the range [-1, 1].
    - In 'fixed' mode, opinions are generated a non - random way using numpy linspace and -lower bound and upper bound from the input parameters

    """

    if mode == "standard":
        opinions = np.random.uniform(low=-1, high=1, size=number_of_media)
        IDs = np.arange(number_of_media)
        media_nodes = [Media(ID, opinion=opinion)
                       for ID, opinion in zip(IDs, opinions)]
        return media_nodes

    elif mode == "uniform":
        opinions = np.random.uniform(
            low=lower_bound, high=upper_bound, size=number_of_media
        )
        IDs = np.arange(number_of_media)
        media_nodes = [Media(ID, opinion=opinion)
                       for ID, opinion in zip(IDs, opinions)]
        return media_nodes

    elif mode == "fixed":
        opinions = np.linspace(
            start=lower_bound, stop=upper_bound, num=number_of_media
        )
        IDs = np.arange(number_of_media)
        media_nodes = [Media(ID, opinion=opinion)
                       for ID, opinion in zip(IDs, opinions)]
        return media_nodes

    elif mode == "gaussian":
        opinions = np.random.normal(loc=mu, scale=sigma, size=number_of_media)
        # Set all values greater than 1 to 1 and all values smaller than -1 to -1
        opinions = np.clip(opinions, -1, 1)
        IDs = np.arange(number_of_media)
        media_nodes = [Media(ID, opinion=opinion)
                       for ID, opinion in zip(IDs, opinions)]
        return media_nodes


def media_conx(network, media, Nc):
    """
    Connect media nodes to random voters in the network.

    Each media node is connected to `Nc` random voters in the network. This function ensures 
    no duplicate connections are made between a media node and a voter. The updated network 
    with media connections is returned.

    Parameters
    ----------
    network : list of list of Voter
        A 2D list representing the voter network, where each element is a `Voter` object.
    media : list of Media
        A list of media nodes, where each media node has an `get_id()` method to retrieve its ID.
    Nc : int
        The number of connections each media node should establish with random voters.

    Returns
    -------
    list of list of Voter
        The updated voter network with media connections established.
    """
    # Get the shape of the network
    n = np.shape(network)

    # Iterate over each media node
    for m in media:
        i = 0
        while i < Nc:  # Make Nc connections for the current media node
            x = random.randint(0, n[0] - 1)  # Random x-coordinate for a voter
            y = random.randint(0, n[1] - 1)  # Random y-coordinate for a voter
            voter = network[x][y]  # Select a random voter

            # Check if the media ID is not already connected to the voter
            if m.get_id() not in voter.get_media_connections():
                # Add the media connection
                voter.add_media_connection(m.get_id())
                i += 1  # Increment the count of connections made for this media node

    return network


def voter_update(voter, h, S, alpha, t0):
    """
    Update the opinion of a voter based on the local field and the current polarization state.

    The opinion threshold to switch between states (red, blue, neutral) depends on the initial threshold `t0`, 
    the total polarization `S`, and the adjustment factor `alpha`. The update favors the party with the minority 
    opinion by lowering its threshold for neutral voters to adopt it.

    Parameters
    ----------
    voter : Voter
        The voter object whose opinion is to be updated. The object must have `get_opinion` and `set_opinion` methods.
    h : float
        The local field experienced by the voter, representing the influence from neighbors and media.
    S : float
        The total polarization in the system. Positive values indicate red majority, and negative values indicate blue majority.
    alpha : float
        Adjustment factor for polarization-dependent threshold shifts.
    t0 : tuple of float
        Initial thresholds for opinion changes:
            - `t0[0]`: Threshold to change from partisan (red/blue) to neutral.
            - `t0[1]`: Threshold to change from neutral to partisan (red/blue).

    Returns
    -------
    int
        Returns `1` if the voter's opinion changes, otherwise `0`.

    Notes
    -----
    - Opinion values are:
        - `1`: Red
        - `0`: Neutral
        - `-1`: Blue
    - Thresholds are adjusted based on the system's polarization, with minority opinions having a lower threshold 
      for adoption by neutral voters.
    """
    changed_opinion = 0
    opinion = voter.get_opinion()

    if S > 0:  # Red majority
        # Adjust thresholds
        t_rn = t0[0]  # Red to neutral (unchanged)
        t_bn = max(min(t0[0] + alpha * S, 0.5), 0)  # Blue to neutral
        t_nr = max(min(t0[1] + alpha * S, 0.5), 0)  # Neutral to red
        t_nb = max(min(t0[1] - alpha * S, 0.5), 0)  # Neutral to blue

        # Update opinions based on the local field
        if opinion == 1 and h < t_rn:  # Red to neutral
            voter.set_opinion(0)
            changed_opinion = 1
        elif opinion == 0:
            if h > t_nr:  # Neutral to red
                voter.set_opinion(1)
                changed_opinion = 1
            elif h < -t_nb:  # Neutral to blue
                voter.set_opinion(-1)
                changed_opinion = 1
        elif opinion == -1 and h > t_bn:  # Blue to neutral
            voter.set_opinion(0)
            changed_opinion = 1

    elif S <= 0:  # Blue majority
        # Adjust thresholds
        t_bn = t0[0]  # Blue to neutral (unchanged)
        t_rn = max(min(t0[0] - alpha * S, 0.5), 0)  # Red to neutral
        t_nr = max(min(t0[1] + alpha * S, 0.5), 0)  # Neutral to red
        t_nb = max(min(t0[1] - alpha * S, 0.5), 0)  # Neutral to blue

        # Update opinions based on the local field
        if opinion == -1 and h > t_bn:  # Blue to neutral
            voter.set_opinion(0)
            changed_opinion = 1
        elif opinion == 0:
            if h > t_nr:  # Neutral to red
                voter.set_opinion(1)
                changed_opinion = 1
            elif h < -t_nb:  # Neutral to blue
                voter.set_opinion(-1)
                changed_opinion = 1
        elif opinion == 1 and h < -t_rn:  # Red to neutral
            voter.set_opinion(0)
            changed_opinion = 1

    return changed_opinion


def local_field(voter, network, media, W):
    """
    Compute the local opinion field experienced by a voter.

    The local field combines the influence of neighboring voters and connected media nodes, 
    weighted by the media authority factor `W`.

    Parameters
    ----------
    voter : Voter
        The voter object whose local field is to be calculated. The object must have methods:
            - `get_neighbors()`: Returns a list of neighboring voter coordinates.
            - `get_number_of_neighbors()`: Returns the total number of neighbors.
            - `get_media_connections()`: Returns a list of IDs of connected media nodes.
    network : list of list of Voter
        The 2D voter network, where each element is a `Voter` object.
    media : list of Media
        A list of media nodes, indexed by their IDs. Each media node must have a `get_opinion()` method.
    W : float
        Media authority factor, which determines the weight of media influence relative to neighbors.

    Returns
    -------
    float
        The local field value, computed as a weighted sum of neighbor and media opinions.

    Notes
    -----
    - The local field influences whether the voter changes their opinion during the update process.
    - Neighbors contribute equally, while media influence is scaled by the authority factor `W`.
    """
    h = 0  # Initialize the local field
    neighbors = voter.get_neighbors()  # Get neighboring voter coordinates
    n = voter.get_number_of_neighbors()  # Number of neighbors
    con_media = voter.get_media_connections()  # List of connected media IDs
    m = len(con_media)  # Number of media connections

    # Sum opinions of neighbors
    for coordinate in neighbors:
        h += network[coordinate[0]][coordinate[1]].get_opinion()

    # Add weighted influence of connected media
    for mid in con_media:
        h += W * media[mid].get_opinion()

    # Normalize by the total influence (neighbors + weighted media)
    return h / (n + W * m)

def get_election_winner(network):
    """
    Returns the party which has relative the majority in the network

    Returns
    -------
    int
        the winning party (can either be 1 or -1)
    """
    opinion_share_df = opinion_share(network)

    # opinion_shared_df[-1][0] gives access to the share of the -1 (blue party)
    # opinion_shared_df[1][0] gives access to the share of the 1 (red party)
    # in case of a tie: 1 (red) wins
    if opinion_share_df[-1][0] > opinion_share_df[1][0]:
        return -1
    else:
        return 1
    
def get_number_of_consecutive_terms(election_results):
    """
    Takes a chronological list of the election results and returns the number of consecutive terms of the ruling party.
    
    Some example outputs: 

    get_number_of_consecutive_terms([1,-1,1,1]) == 1
    get_number_of_consecutive_terms([1,-1,-1,-1,-1]) == 3

    Parameters
    ----------
    election_results : list

    Returns:
    number_of_consecutive_terms : int
    """

    if not election_results:  # Handle empty list
        return 0
    
    last_value = election_results[-1]

    count = 0
    for num in reversed(election_results):
        if num == last_value:
            count += 1
        else:
            break

    count -= 1 # correct the counting such that [1,-1,1,1] for example results in 1 and not 2
    return count


def network_update(network, media, Nv, W, t0, alpha, mfeedback):
    """
    Update the network for one timestep by randomly selecting voters and updating their opinions.

    The function randomly picks `Nv` voters from the network, computes their local opinion field, 
    and updates their opinions based on polarization and thresholds. Optionally, applies media feedback 
    to the selected voters.

    Parameters
    ----------
    network : list of list of Voter
        The 2D voter network, where each element is a `Voter` object.
    media : list
        A list of media nodes, indexed by their IDs. Each media node must have a `get_opinion()` method.
    Nv : int
        The number of voters to randomly select and update during this timestep.
    W : float
        Media authority factor, which determines the weight of media influence relative to neighbors.
    t0 : tuple of float
        Initial thresholds for opinion changes:
            - `t0[0]`: Threshold to change from partisan (red/blue) to neutral.
            - `t0[1]`: Threshold to change from neutral to partisan (red/blue).
    alpha : float
        Adjustment factor for polarization-dependent threshold shifts.
    mfeedback : bool
        If `True`, applies media feedback to voters after their opinion update.

    Returns
    -------
    int
        The number of voters whose opinions changed during this timestep.

    Notes
    -----
    - The function assumes the existence of `local_field`, `polarization`, and `voter_update` functions:
        - `local_field(voter, network, media, W)` computes the local opinion field.
        - `polarization(network)` computes the current system polarization.
        - `voter_update(voter, h, S, alpha, t0)` updates a voter's opinion based on the local field and polarization.
    - If `mfeedback` is enabled, voters apply media feedback via their `media_feedback` method.
    """
    changed_voters = 0  # Counter for voters who changed opinions

    # Network dimensions
    n = np.shape(network)

    for _ in range(Nv):  # Randomly pick and update Nv voters
        x = random.randint(0, n[0] - 1)  # Random x-coordinate
        y = random.randint(0, n[1] - 1)  # Random y-coordinate
        voter = network[x][y]  # Selected voter

        # Compute the local opinion field
        h = local_field(voter, network, media, W)

        # Compute the current polarization of the system
        s = polarization(network)

        # Update the voter's opinion and track changes
        changed_voters += voter_update(voter, h, s, alpha, t0)

        # Apply media feedback if enabled
        if mfeedback:
            voter.media_feedback(media)

    return changed_voters



def update_media(days, media, election_results, initial_media_opinion, number_of_days_election_cycle, media_update_cycle=1):
    #should all voters get updated?
    if days % media_update_cycle == 1:  # Sk ← Sk + E. (I was here, but thats wrong)#maybe check if it is checked every day,
        media_change = random.normal(0, 0.5)  + initial_media_opinion
        for i,_ in enumerate(media):
            new_opinion = media[i].get_opinion() + media_change
            if abs(new_opinion) < 1: #why here 0.5?
                media[i].set_opinion(new_opinion)
            elif new_opinion < 0:
                media[i].set_opinion(-1)
                #print(f"media opinion is too low: {media[i].get_opinion()+media_change}")
            elif new_opinion > 0:
                media[i].set_opinion(1)
                #print(f"media opinion is too high: {media[i].get_opinion()+media_change}")


    if days % number_of_days_election_cycle == 1:# Sk ← Sk + 0.376 × DUR*I (I=who is in power)
        dur=0
        if get_number_of_consecutive_terms(election_results) >= 2:
            dur = 1 + (get_number_of_consecutive_terms(election_results) - 2) * 0.25  # is it 0.1 or
            dur = min(dur, 1.5)
        if election_results:  # Check if the list is not empty
            i = (-1)*election_results[-1] if election_results[-1] is not None else 0
        else:
            i = 0  # Default value if the list is empty
        media_change = dur * 0.00376 * i

        #if media_change * election_results[-1] > 0:#als assert schrieben?
            #print(f"alarm, media_change supports election winner: media_change:{media_change}, election winner: {election_results[-1]}")
        for i,_ in enumerate(media):
            new_opinion = media[i].get_opinion()+media_change
            if abs(new_opinion)<1: # why here 0.5?
                media[i].set_opinion(new_opinion)
            elif new_opinion < 0:
                media[i].set_opinion(-1)
                #print(f"media opinion is too low: {media[i].get_opinion()+media_change}")
            elif new_opinion > 0:
                media[i].set_opinion(1)
                #print(f"media opinion is too high: {media[i].get_opinion()+media_change}")
    return media

