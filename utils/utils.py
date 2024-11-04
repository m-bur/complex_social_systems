
import numpy as np
import pandas as pd

class Voter:
    def __init__(self, i, j, opinion=0):
        """
        Initialize a Voter object.

        Parameters:
        i (int): The row coordinate of the voter.
        j (int): The column coordinate of the voter.
        opinion (int): The opinion of the voter, which can be -1, 0, or 1. Default is 0.
        """
        self.i = i
        self.j = j
        self.opinion = opinion
        self.neighbors = []  # This will hold the coordinates of neighboring voters

    def add_neighbor(self, neighbor_i, neighbor_j):
        """
        Adds a neighbor's coordinates to the list of neighbors.

        Parameters:
        neighbor_i (int): The row coordinate of the neighbor.
        neighbor_j (int): The column coordinate of the neighbor.
        """
        self.neighbors.append((neighbor_i, neighbor_j))

    def set_opinion(self, opinion):
        """
        Sets the voter's opinion.

        Parameters:
        opinion (int): The new opinion of the voter, which should be -1, 0, or 1.
        """
        if opinion in [-1, 0, 1]:
            self.opinion = opinion
        else:
            raise ValueError("Opinion must be -1, 0, or 1")

    def get_neighbors(self):
        """
        Returns the list of neighbors' coordinates.

        Returns:
        list: A list of tuples representing the coordinates of neighboring voters.
        """
        return self.neighbors
    
    def get_number_of_neighbors(self):
        """
        Returns the list of neighbors' coordinates.

        Returns:
        list: A list of tuples representing the coordinates of neighboring voters.
        """
        return len(self.neighbors)

    def __repr__(self):
        """
        Returns a string representation of the Voter object for debugging purposes.
        """
        return f"Voter(i={self.i}, j={self.j}, opinion={self.opinion}, neighbors={self.neighbors})"
    

