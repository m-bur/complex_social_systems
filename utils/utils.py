import numpy as np
import pandas as pd


class Voter:
    def __init__(self, i, j, opinion=0):
        """
        Initialize a Voter object.

        Parameters
        ----------
        i : int
            The row coordinate of the voter.
        j : int
            The column coordinate of the voter.
        opinion : int, optional
            The opinion of the voter, which can be -1, 0, or 1. Default is 0.
        """
        self.i = i
        self.j = j
        self.opinion = opinion
        self.neighbors = []  # This will hold the coordinates of neighboring voters
        self.media_connections = []  # List to store IDs of connected media nodes

    def add_neighbor(self, neighbor_i, neighbor_j):
        """
        Adds a neighbor's coordinates to the list of neighbors.

        Parameters
        ----------
        neighbor_i : int
            The row coordinate of the neighbor.
        neighbor_j : int
            The column coordinate of the neighbor.
        """
        self.neighbors.append((neighbor_i, neighbor_j))

    def set_opinion(self, opinion):
        """
        Sets the voter's opinion.

        Parameters
        ----------
        opinion : int
            The new opinion of the voter, which should be -1, 0, or 1.

        Raises
        ------
        ValueError
            If the opinion is not -1, 0, or 1.
        """
        if opinion in [-1, 0, 1]:
            self.opinion = opinion
        else:
            raise ValueError("Opinion must be -1, 0, or 1")

    def add_media_connection(self, media_id):
        """
        Connects the voter to a media node by adding the media_id to the media_connections list.

        Parameters
        ----------
        media_id : int or str
            A unique identifier for the media node.
        """
        self.media_connections.append(media_id)

    def remove_media_connection(self, media_id):
        """
        Removes a media connection by removing the media_id from the media_connections list.

        Parameters
        ----------
        media_id : int or str
            The unique identifier of the media node to disconnect.

        Raises
        ------
        ValueError
            If the media_id is not found in the media_connections list.
        """
        try:
            self.media_connections.remove(media_id)
        except ValueError:
            raise ValueError(f"Media ID {media_id} is not connected to this voter.")

    def get_neighbors(self):
        """
        Returns the list of neighbors' coordinates.

        Returns
        -------
        list of tuple
            A list of tuples representing the coordinates of neighboring voters.
        """
        return self.neighbors

    def get_number_of_neighbors(self):
        """
        Returns the number of neighbors.

        Returns
        -------
        int
            The number of neighboring voters.
        """
        return len(self.neighbors)

    def get_media_connections(self):
        """
        Returns the list of connected media node IDs.

        Returns
        -------
        list
            A list of media node IDs to which the voter is connected.
        """
        return self.media_connections

    def __repr__(self):
        """
        Returns a string representation of the Voter object for debugging purposes.

        Returns
        -------
        str
            A string representation of the Voter object.
        """
        return f"Voter(i={self.i}, j={self.j}, opinion={self.opinion}, neighbors={self.neighbors}, media_connections={self.media_connections})"


class Media:
    def __init__(self, media_id, opinion=0.0):
        """
        Initialize a Media object.

        Parameters
        ----------
        media_id : int or str
            A unique identifier for the media node.
        opinion : float, optional
            The opinion of the media, which should be a value between -1 and 1. Default is 0.0.

        Raises
        ------
        ValueError
            If the opinion is not within the range [-1, 1].
        """
        self.media_id = media_id
        self.set_opinion(opinion)  # Ensures opinion is within the valid range

    def set_opinion(self, opinion):
        """
        Sets the media's opinion.

        Parameters
        ----------
        opinion : float
            The new opinion of the media, which should be between -1 and 1.

        Raises
        ------
        ValueError
            If the opinion is not within the range [-1, 1].
        """
        if -1.0 <= opinion <= 1.0:
            self.opinion = opinion
        else:
            raise ValueError("Opinion must be between -1 and 1 (inclusive).")

    def get_opinion(self):
        """
        Returns the opinion of the media.

        Returns
        -------
        float
            The opinion of the media.
        """
        return self.opinion

    def get_id(self):
        """
        Returns the unique identifier of the media.

        Returns
        -------
        int or str
            The unique identifier of the media node.
        """
        return self.media_id

    def __repr__(self):
        """
        Returns a string representation of the Media object for debugging purposes.

        Returns
        -------
        str
            A string representation of the Media object.
        """
        return f"Media(id={self.media_id}, opinion={self.opinion})"
