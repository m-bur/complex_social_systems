import random
import numpy as np
import pandas as pd
import random
from utils.measure import polarization


def init_df_conx(c_min, c_max, gamma, L):
    """ Initialize the connection dataframe, number of rows is L * L
        columns = x; y; number of connections; numbers left (auxiliary); connection list
    :param c_min: minimal number of connections
    :param c_max: maximal number of connections
    :param gamma: distribution parameter for the number of connections (higher gamma -> lower connections)
    :param L: L: Side length, i.e., the total number of nodes N = L * L
    :return: a dataframe of the connections
    """
    # Get the probability list between c_min and c_max
    prob_list = []
    for k in range(c_min, c_max + 1):
        prob = k ** (-gamma)
        prob_list.append(prob)
    prob_sum = sum(prob_list)
    prob_list_norm = [_ / prob_sum for _ in prob_list] + [1]  # normalized prob list
    prob_list = [sum(prob_list_norm[:_]) for _ in range(len(prob_list_norm))]  # accumulative prob list

    # Randomly assign number of connections for the population
    df_conx = pd.DataFrame(columns=['x', 'y', 'num', 'num_left', 'connection'])
    for i in range(L):
        for j in range(L):
            # Store the row with 'x', 'y', 'number connections', and 'connection list'
            val = random.random()
            index = next(_ for _ in range(len(prob_list)) if prob_list[_] <= val < prob_list[_ + 1]) + c_min
            df_conx.loc[len(df_conx)] = [i, j, index, index, []]
    return df_conx


def prob_first_connection(dist, L, L_G):
    """ Probability function to determine whether the first level connection exists between two nodes
    :param dist: cartesian distance between two nodes
    :param L: Side length, i.e., the total number of nodes N = L * L
    :param L_G: Side length of a local group
    :return: Probability of a link (before normalization)
    """
    a = L_G
    b = a / 4
    prob = 1 / (1 + np.exp((dist - a) / b)) + 0.001 * (L - dist) / L
    if prob < 0:
        prob = 0
    return prob


def calc_first_conx_matrix(L, L_G):
    """ Generate the first level connection matrix
    :param L: Side length, i.e., the total number of nodes N = L * L
    :param L_G: Side length of a local group
    :return: the probability matrix of the first level connection
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
    """ Generate the second level connection matrix
    :param L: Side length, i.e., the total number of nodes N = L * L
    :param first_conx_matrix: the first level connection matrix
    :param p_c: probability parameter for a second connection
    :return: the probability matrix of the second level connection
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
    """ Generate the connection dataframe, i.e., network
    :param L: Side length
    :param df_conx: connection dataframe
    :param connection_matrix: the sum of fist and second level connection matrix
    :return:
    """
    print("==== Generate network")
    pop = L * L
    connection_matrix_norm = np.zeros((pop, pop))
    for p in range(pop):
        prob_list = connection_matrix[p, :]
        num_left = df_conx['num_left']
        prob_list = [prob_list[_] * num_left[_] for _ in range(pop)]
        sum_prob = np.sum(prob_list)
        if sum_prob > 0:
            prob_list = [_ / sum_prob for _ in prob_list]

            i = p // L
            j = p % L
            sample = df_conx.loc[p, 'num_left']
            left_sample = len([_ for _ in prob_list if _ != 0])
            if sample > left_sample:
                sample = left_sample
            selected_idx = list(np.random.choice(len(prob_list), size=sample, replace=False, p=prob_list))

            for q in selected_idx:
                m = q // L
                n = q % L
                df_conx.at[p, 'connection'].append((m, n))
                df_conx.at[p, 'num_left'] = df_conx.at[p, 'num_left'] - 1
                df_conx.at[q, 'connection'].append((i, j))
                df_conx.at[q, 'num_left'] = df_conx.at[q, 'num_left'] - 1
                connection_matrix_norm[p, q] = 0
                connection_matrix_norm[q, p] = 0

    df_conx = df_conx[['x', 'y', 'num', 'connection']]
    return df_conx



def init_network(df_conx, network):
    """
    Initiliaize the voter network with given connections. Returns the list of network nodes with neighbors according to df_conx.
    """
    for i in range(df_conx.shape[0]):
        row = df_conx.iloc[i]
        neighbors = row["connection"]
        voter = network[row["x"]][row["y"]]
        voter.set_opinion(random.choice([-1,0,1]))
        for ncoordinates in neighbors:
            voter.add_neighbor(ncoordinates[0], ncoordinates[1])
    return network



def init_media(mu, sigma, media):
    """
    Initialize the media network. Opionion follows countinuous uniform probability distribution.
    """
    for m in media:
        m.set_opinion(np.clip(random.gauss(mu, sigma), -1 ,1))
    return media


def media_conx(network, media, Nc):
    """
    Make connections between Media and voters. Each media node is connected to Nc random voters. Return the updated voter network with media connections.
    """
    n = np.shape(network)
    for m in media:
        i = 0
        while i < Nc:                          # make Nc connections per media node
            x = random.randint(0, n[0] - 1)    # random x voter value
            y = random.randint(0, n[1] - 1)    # random y voter value
            voter = network[x][y]
            if m.get_id() not in voter.get_media_connections():     # make sure that there are no double connections
                voter.add_media_connection(m.get_id())
                i += 1
    return network


def voter_update(voter, h, S, alpha, t0):
    """
    Update the opinion of the voters based on the local field h. The threshold to change the opinion depends on the initial threshold t0 and the total polarization S.
    It favors the party which has the minority.
    """
    opinion = voter.get_opinion()
    if S > 0:                                       # if red has the majority
        t_rn = t0[0]                                # threshold to change from red to neutral (unchanged)
        t_bn = max(min(t0[0] + alpha*S, 0.5), 0)    # threshold to change from blue to neutral (higher, less likely)
        t_nr = max(min(t0[1] + alpha*S, 0.5), 0)    # threshold to change from neutral to red (higher, less likely)
        t_nb = max(min(t0[1] - alpha*S, 0.5), 0)    # threshold to change from neutral to blue (lower, more likely)
        if opinion == 1 and h < t_rn:
            voter.set_opinion(0)
        elif opinion == 0:
            if h > t_nr:
                voter.set_opinion(1)
            elif h < -t_nb:
                voter.set_opinion(-1)
        elif opinion == -1 and h > t_bn:
            voter.set_opinion(0)
    if S <= 0:                                      # if blue has the majority
        t_bn = t0[0]                                # threshold to change from blue to neutral (unchanged)
        t_rn = max(min(t0[0] - alpha*S, 0.5), 0)    # threshold to change from red to neutral (higher, less likely)
        t_nr = max(min(t0[1] + alpha*S, 0.5), 0)    # threshold to change from neutral to red (lower, more likely)
        t_nb = max(min(t0[1] - alpha*S, 0.5), 0)    # threshold to change from neutral to blue (higher, less likely)
        if opinion == 1 and h < -t_rn:
            voter.set_opinion(0)
        elif opinion == 0:
            if h > t_nr:
                voter.set_opinion(1)
            elif h < -t_nb:
                voter.set_opinion(-1)
        elif opinion == -1 and h > t_bn:
            voter.set_opinion(0)
    return voter


def local_field(voter, network, media, W):
    """
    Computes the local opinion field for the voter. The media authority is W 
    """
    h = 0
    neighbors = voter.get_neighbors()
    n = voter.get_number_of_neighbors()
    con_media = voter.get_media_connections()
    m = len(con_media)
    for coordinate in neighbors:
        h += network[coordinate[0]][coordinate[1]].get_opinion()
    for mid in con_media:
        h += W*media[mid].get_opinion()
    return h /(n + W*m)


def network_update(network, media, Nv, W, t0, alpha):
    """
    Update the network one timestep by randomly picking Nv voters and updating their opinion. Media authority W and initial thresholds t0 with parameter alpha.
    """
    for _ in range(Nv):
        n = np.shape(network)
        x = random.randint(0, n[0] - 1)    # random x voter value
        y = random.randint(0, n[1] - 1)    # random y voter value
        voter = network[x][y]
        h = local_field(voter, network, media, W)
        s = polarization(network)
        voter_update(voter, h, s, alpha, t0)
    return network
