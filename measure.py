import numpy as np


def polarization(network):
    """
    Returns the average opinion of  the network.
    """
    S = 0
    n = np.shape(network)
    for i in range(n[0]):
        for j in range(n[1]):
            S += network[i][j].get_opinion()
    return S / (n[0]*n[1])


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