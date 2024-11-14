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
        media_feedback_probability=0.5,
        meadia_feedback_threshold_replacement_neutral=0.1,
    ):
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

    def get_opinion_of_media(self, media_df):
        """
        This functions returns a list of all the opinions of nodes in the voter network
        with which the voter is connected.

        Parameters
        ----------
        network : pd.DataFrame
            A DataFrame which contains all the media nodes and their media_ids

        Returns
        -------
        list
            A list that contains the opinion of all the connected media.

        """

        res = [
            media_df.loc[media_id, "node"].get_opinion() for media_id in self.get_media_connections()
        ]
        return res

    def media_feedback(self, media_df):
        """
        This is the algorithm for media feedback as defined in the paper.


        """

        if self.get_opinion() == -1:
            for media_id in self.media_connections:
                p = np.random.uniform(0, 1)  # generate a random number between 0 and 1

                if (
                    media_df.loc[media_id, "node"].get_opinion() > 0
                    and p < self.media_feedback_probability
                ):

                    # check if there are any media nodes left that are not yet connected to this voter
                    if len(media_df) > len(self.media_connections):
                        # get all the media nodes that are not connected to this voter in a new dataframe
                        unconnected_media_df = media_df.loc[
                            ~media_df["media_id"].isin(self.media_connections)
                        ]

                        # generate a random integer to pick a random node:
                        i = np.random.randint(0, len(unconnected_media_df) - 1)

                        new_media_id = unconnected_media_df["media_id"].iloc[i]

                        # if the randomly chosen new
                        if media_df.loc[new_media_id, "node"].get_opinion() <= 0:
                            self.remove_media_connection(media_id)
                            self.add_media_connection(new_media_id)

        elif self.get_opinion() == 1:
            pass
        elif self.get_opinion() == 0:
            pass

        return

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


def generate_media_landscape(
    number_of_media, mode="standard", mu=0, sigma=0.25, lower_bound=-1, upper_bound=1
):
    """
    generate a pandas dataframe which contains the medianodes and their ids
    """

    if mode == "standard":
        opinions = np.random.uniform(low=-1, high=1, size=number_of_media)
        IDs = np.arange(number_of_media) + 10
        media_nodes = [Media(ID, opinion=opinion) for ID, opinion in zip(IDs, opinions)]
        df = pd.DataFrame({"media_id": IDs, "node": media_nodes})
        df.set_index("media_id", inplace=True)  # Set media_id as the index
        df['media_id'] = df.index
        return df

    elif mode == "uniform":
        opinions = np.random.uniform(
            low=-lower_bound, high=upper_bound, size=number_of_media
        )
        IDs = np.arange(number_of_media)
        media_nodes = [Media(ID, opinion=opinion) for ID, opinion in zip(IDs, opinions)]
        df = pd.DataFrame({"media_id": IDs, "node": media_nodes})
        return df

    elif mode == "gaussian":
        opinions = np.random.normal(loc=mu, scale=sigma, size=number_of_media)
        # Set all values greater than 1 to 1 and all values smaller than -1 to -1
        opinions = np.clip(opinions, -1, 1)
        IDs = np.arange(number_of_media)
        media_nodes = [Media(ID, opinion=opinion) for ID, opinion in zip(IDs, opinions)]
        df = pd.DataFrame({"media_id": IDs, "node": media_nodes})
        return df
