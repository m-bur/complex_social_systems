import numpy as np
import pandas as pd


class Voter:
    def __init__(
        self,
        i,
        j,
        opinion=0,
        media_weight=10,
        media_feedback_turned_on=False,
        media_feedback_probability=0.1,
        meadia_feedback_threshold_replacement_neutral=0.1,
    ):
        """
        Initialize a Voter object, representing a voter with specific coordinates, opinion, 
        and media-related attributes, including feedback mechanisms and media connections.

        Parameters
        ----------
        i : int
            The row coordinate of the voter in the grid or network.
        j : int
            The column coordinate of the voter in the grid or network.
        opinion : int, optional
            The initial opinion of the voter, which can be -1 (against), 0 (neutral), or 1 (supportive).
            Default is 0 (neutral).
        media_weight : int, optional
            A weight factor indicating the importance of media in influencing the voter's opinion 
            relative to neighboring voters. Default is 10.
        media_feedback_turned_on : bool, optional
            A flag indicating whether media feedback is active. If `True`, the voter can be influenced 
            by media nodes. Default is `False`.
        media_feedback_probability : float, optional
            The probability that the voter will seek to change media they disagree with. (When media feedback is enabled.) 
            This corresponds to the parameter beta in the associated paper (Media preference increases polarization in an agent-based
            election model by Di Benedetto et al.). Default is 0.5.
        meadia_feedback_threshold_replacement_neutral : float, optional
            The threshold that defines the range of media opinions considered neutral for the voter with opinion 0.
            Only media opinions within this range will be accepted, otherwise the voter will seek to replace the media node. Default is 0.1.

        Attributes
        ----------
        i : int
            The row coordinate of the voter.
        j : int
            The column coordinate of the voter.
        opinion : int
            The voter's current opinion (-1, 0, or 1).
        neighbors : list of tuple of int
            A list to hold the coordinates of neighboring voters.
        media_connections : list of int
            A list to store IDs of connected media nodes influencing the voter's opinion.
        media_weight : int
            The weight by which media influences the voter's opinion relative to other voters.
        media_feedback_turned_on : bool
            A flag indicating whether the voter can be influenced by media feedback.
        media_feedback_probability : float
            The probability that the voter will cut ties with media nodes they disagree with, feedback when activated.
        meadia_feedback_threshold_replacement_neutral : float
            The threshold for accepting new media opinions if the voter is neutral.
        """
        self.i = i
        self.j = j
        self.opinion = opinion
        self.neighbors = []  # This will hold the coordinates of neighboring voters
        self.media_connections = []  # List to store IDs of connected media nodes
        self.media_weight = media_weight  # The factor by which is media is weighted higher than a normal neighbor

        # boolean to switch on and off the media feedback
        self.media_feedback_turned_on = media_feedback_turned_on

        # this value defines the probability to look for a new medium,
        # --> this corresponds to the parameter beta in the paper
        self.media_feedback_probability = media_feedback_probability

        # the neutral voter only accepts new media with opinion in the range +/- this threshold
        self.meadia_feedback_threshold_replacement_neutral = (
            meadia_feedback_threshold_replacement_neutral
        )

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

    def get_opinion_of_neighbours(self, network):
        """
        This functions returns a list of all the opinions of nodes in the voter network
        with which the voter is connected.

        Parameters
        ----------
        network : 2d np.array
            The matrix on which all the voter nodes are placed.

        Returns
        -------
        list
            A list that contains the opinion of all the neighbours.

        """

        res = [
            network[neighbour_coordinates[0], neighbour_coordinates[1]].get_opinion()
            for neighbour_coordinates in self.get_neighbors()
        ]
        return res

    def get_opinion_of_media(self, media):
        """
        This functions returns a list of all the opinions of nodes in the voter network
        with which the voter is connected.

        Parameters
        ----------
        media: list
            A list that contains all the media nodes

        Returns
        -------
        list
            A list that contains the opinion of all the connected media.

        """

        res = [
            media[media_id].get_opinion()
            for media_id in self.get_media_connections()
        ]
        return res

    def get_weighted_average_opinion_of_enviroment(self, network, media):
        """
        This function calculates the weighted average of the opinion of the connected voter and media nodes.

        Parameters
        ----------
        network : 2d np.array
            The matrix on which all the voter nodes are placed.
        media: list
            A list that contains all the media nodes

        Returns
        -------
        float
            the weighted average of the opinion of nodes that are connected to this voter.
        """
        neighbor_opinions = self.get_opinion_of_neighbours(network)
        media_opinions = self.get_opinion_of_media(media)

        weighted_average = (
            sum(neighbor_opinions) + self.media_weight * sum(media_opinions)
        ) / (len(neighbor_opinions) + self.media_weight * len(media_opinions))

        return weighted_average

    def media_feedback(self, media):
        """
        This is the algorithm for media feedback as defined in the paper.

        Parameters
        ----------
        media: list
            A list that contains all the media nodes
        """

        if self.get_opinion() == -1:
            for media_id in self.media_connections:
                p = np.random.uniform(0, 1)  # generate a random number between 0 and 1

                if (
                    media[media_id].get_opinion() > 0
                    and p < self.media_feedback_probability
                ):

                    # check if there are any media nodes left that are not yet connected to this voter
                    if len(media) > len(self.media_connections):
                        # get all the media nodes that are not connected to this voter in a new dataframe
                        unconnected_media = [media[media_id] for media_id in range(len(media)) if media_id not in self.media_connections]

                        # generate a random integer to pick a random node:
                        i = np.random.randint(0, len(unconnected_media) - 1)

                        # if the randomly chosen new
                        if unconnected_media[i].get_opinion() <= 0:
                            self.remove_media_connection(media_id)
                            self.add_media_connection(unconnected_media[i].get_id())

        elif self.get_opinion() == 1:
            for media_id in self.media_connections:
                p = np.random.uniform(0, 1)  # generate a random number between 0 and 1

                if (
                    media[media_id].get_opinion() < 0
                    and p < self.media_feedback_probability
                ):

                    # check if there are any media nodes left that are not yet connected to this voter
                    if len(media) > len(self.media_connections):
                        # get all the media nodes that are not connected to this voter in a new dataframe
                        unconnected_media = [media[media_id] for media_id in range(len(media)) if media_id not in self.media_connections]

                        # generate a random integer to pick a random node:
                        i = np.random.randint(0, len(unconnected_media) - 1)

                        # if the randomly chosen new
                        if unconnected_media[i].get_opinion() >= 0:
                            self.remove_media_connection(media_id)
                            self.add_media_connection(unconnected_media[i].get_id())

        elif self.get_opinion() == 0:
            for media_id in self.media_connections:
                p = np.random.uniform(0, 1)  # generate a random number between 0 and 1

                if (
                    abs(media[media_id].get_opinion())
                    > self.meadia_feedback_threshold_replacement_neutral
                    and p < self.media_feedback_probability
                ):

                    # check if there are any media nodes left that are not yet connected to this voter
                    if len(media) > len(self.media_connections):
                        # get all the media nodes that are not connected to this voter in a new dataframe
                        unconnected_media = [media[media_id] for media_id in range(len(media)) if media_id not in self.media_connections]

                        # generate a random integer to pick a random node:
                        i = np.random.randint(0, len(unconnected_media) - 1)

                        # if the randomly chosen new
                        if (
                            abs(unconnected_media[i].get_opinion())
                            <= self.meadia_feedback_threshold_replacement_neutral
                        ):
                            self.remove_media_connection(media_id)
                            self.add_media_connection(unconnected_media[i].get_id())

    def get_neighbors(self):
        """
        Returns the list of neighbors' coordinates.

        Returns
        -------
        list of tuple
            A list of tuples representing the coordinates of neighboring voters.
        """
        return self.neighbors

    def get_opinion(self):
        """
        Returns the opinion.

        Returns
        -------
        int
            The opinion of the voters.
        """
        return self.opinion

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
        return (
            f"Voter(i={self.i}, j={self.j}, opinion={self.opinion}, "
            f"neighbors={self.neighbors}, media_connections={self.media_connections})"
        )


class Media:
    def __init__(self, media_id, opinion=0.0, category_threshold=1/3):
        """
        Initialize a Media object.

        Parameters
        ----------
        media_id : int or str
            A unique identifier for the media node.
        opinion : float, optional
            The opinion of the media, which should be a value between -1 and 1. Default is 0.0.
        category_threshold : float
            Opinion values inside the interval [t,-t] are categorized as "neutral". Values bigger than 
            the value "red", otherwise "blue". 

        Raises
        ------
        ValueError
            If the opinion is not within the range [-1, 1].
        """
        self.category_threshold = category_threshold
        self.media_id = media_id
        self.opinion = opinion
        self.set_opinion(opinion)  # Ensures opinion is within the valid range

    def update_category(self):
        """updates the category of the media opinion"""

        if self.opinion > self.category_threshold:
            self.category = "red"
        elif self.opinion < -self.category_threshold:
            self.category = "blue"
        else:
            self.category = "neutral"

    def get_category(self):
        """
        Returns the opinion category of the medium.

        Returns
        -------
        string
            The opinion category of the medium. This can be "blue", "neutral" or "red".
        """
        return self.category

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
            # do not forget to update the category, needs to be changed with opinion
            self.update_category()
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
        return f"Media(id={self.media_id}, opinion={self.opinion}, category={self.category})"
