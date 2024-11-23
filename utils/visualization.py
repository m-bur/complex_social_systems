import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from matplotlib.colors import ListedColormap, BoundaryNorm
import imageio


def visualize_matrix(matrix, output_folder, filename=None):
    """
    Visualizes a matrix with values -1 (blue), 0 (grey), 1 (red) and saves it as an image.

    Parameters
    ----------
    matrix : numpy.ndarray
        A 2D array of integers containing values -1, 0, or 1.
    output_folder : str
        Path to the folder where the output image file will be saved.
    filename : str, optional
        Name of the output image file. If None, a unique filename based on
        the current timestamp is generated.

    Returns
    -------
    None
        The function saves the image file but does not return any value.
    """

    # Create a color map: red for -1, grey for 0, blue for 1
    cmap = ListedColormap(['blue', 'grey', 'red'])
    bounds = [-1.5, -0.5, 0.5, 1.5]  # Boundaries for the values
    norm = BoundaryNorm(bounds, cmap.N)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate a unique filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"matrix_visualization_{timestamp}.png"

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest")
    ax.axis("off")  # Remove axes for a cleaner image

    # Save the image
    output_path = os.path.join(output_folder, filename)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)

    plt.close(fig)


def visualize_network(network, folder, filename):
    """
    Visualize the voter network by plotting the opinion distribution across the nodes.

    Parameters
    ----------
    network : array-like
        A 2D array representing the network, where each element corresponds to a node (voter) in the network.
        Each node should have a method `get_opinion()` that returns the opinion of that voter.
    folder : str
        The folder path where the visualized network will be saved.
    filename : str
        The name of the file where the visualized network will be saved.

    Notes
    -----
    This function constructs a 2D matrix representing the opinions of each voter in the network,
    and then calls the `visualize_matrix` function to generate and save the visualization.
    """
    # Get the shape of the network (number of rows and columns)
    n = np.shape(network)

    # Initialize a matrix to store the opinion values
    matrix = np.zeros((n[0], n[1]), dtype=int)

    # Loop through each node in the network and assign its opinion to the matrix
    for i in range(n[0]):
        for j in range(n[1]):
            # Get the opinion of the current node and store it in the matrix
            matrix[i, j] = network[i][j].get_opinion()

    # Visualize the matrix of opinions using the visualize_matrix function
    visualize_matrix(matrix, folder, filename)


def visualize_network_evolution(networks, output_folder, gif_filename="network_evolution.gif"):
    """
    Visualizes the evolution of a network over time and saves the animation as a GIF.

    Parameters
    ----------
    networks : list of list of Voter
        A list of time steps, where each time step is a list of lists representing the network at that time.
        Each element in the network is an instance of the Voter class.
    output_folder : str
        Path to the folder where the output images and the GIF will be saved.
    gif_filename : str, optional
        The filename for the generated GIF. Default is "network_evolution.gif".

    Returns
    -------
    None
        The function saves the GIF to the specified output folder but does not return any value.

    Notes
    -----
    The function generates a color map for the network visualization:
    - -1 is shown in blue
    - 0 is shown in grey
    - 1 is shown in red
    """

    # Create a color map: blue for -1, grey for 0, red for 1
    cmap = ListedColormap(['blue', 'grey', 'red'])

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create a list to store the individual frames for the GIF
    frames = []

    # Loop over the networks at different time steps and create frames
    for t, network in enumerate(networks):
        # Get the opinions of all voters in the network at time step t
        matrix = np.array([[voter.get_opinion()
                          for voter in row] for row in network])

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.imshow(matrix, cmap=cmap, interpolation="nearest")
        ax.axis("off")  # Hide axes for better visualization

        # Construct the filename for the frame
        frame_filename = os.path.join(output_folder, f"frame_{t}.png")
        fig.savefig(frame_filename, bbox_inches="tight", pad_inches=0)

        # Add the frame to the frames list
        frames.append(imageio.imread(frame_filename))

        # Close the figure to free up memory
        plt.close(fig)

        # delete the individual frame image after reading it
        os.remove(frame_filename)

    # Save the frames as a GIF
    gif_path = os.path.join(output_folder, gif_filename)
    imageio.mimsave(gif_path, frames, duration=0.75)  # 0.75 seconds per frame
    
    
def opinion_trend_single_frame(op_trend, time_step):
    """
    Generate a single frame for the opinion trend plot at a given time step.

    Parameters
    ----------
    op_trend : pandas.DataFrame
        DataFrame where columns represent different opinions and the index is time.
    time_step : int
        The current time step to plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    fig, ax = plt.subplots()
    for column in op_trend.columns:
        color = 'blue' if column == -1 else 'grey' if column == 0 else 'red'
        xs = op_trend.index[:time_step].to_numpy()
        ys = op_trend[column][:time_step]
        ax.plot(xs, ys, 
                label=f"Opinion {column}", color=color)
        # Highlight the current point with a large open circle
        ax.scatter(xs[-1], 
                   ys.iloc[-1], 
                   color=color, edgecolor='black', s=100, zorder=5)

        # Draw a dashed line to the x-axis
        ax.axvline(xs[-1], color='k', linestyle="--", alpha=0.6)
    # Configure the plot labels and limits
    ax.set_xlabel("$t [\\mathrm{d}]$", fontsize=16)
    ax.set_ylabel("Opinion Share", fontsize=16)
    ax.set_xlim([0, op_trend.index.max()])  # x-axis range based on time
    y_max = 1.3 * op_trend.max().max()  # Global maximum of op_trend
    ax.set_ylim([0, y_max])  # Set y-axis limits dynamically based on the maximum value
    # Turn on the grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper right", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increase major tick label size
    fig.subplots_adjust(left=0.075, right=1, top=1, bottom=0.125)
    return fig


def network_evolution_single_frame(network, cmap):
    """
    Generate a single frame for the network visualization at a given time step.

    Parameters
    ----------
    network : list of list of Voter
        The network at the current time step.
    cmap : matplotlib.colors.ListedColormap
        The color map for the visualization.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    matrix = np.array([[voter.get_opinion() for voter in row] for row in network])
    ax.imshow(matrix, cmap=cmap, interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig

def combined_visualization(op_trend, networks, output_folder, gif_filename="combined_evolution.gif"):
    """
    Combines the opinion trend plot and network evolution visualization into a single GIF.

    Parameters
    ----------
    op_trend : pandas.DataFrame
        Opinion trend data.
    networks : list of list of Voter
        Network data over time.
    output_folder : str
        Folder to save the GIF.
    gif_filename : str
        Name of the output GIF file.

    Returns
    -------
    None
    """
    os.makedirs(output_folder, exist_ok=True)
    cmap = ListedColormap(['blue', 'grey', 'red'])
    frames = []

    for t, network in enumerate(networks):
        # Generate the two plots
        trend_fig = opinion_trend_single_frame(op_trend, t+1)
        network_fig = network_evolution_single_frame(network, cmap)

        # Combine the two plots into one figure
        combined_fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'wspace': 0})  # Reduce spacing
        trend_canvas = trend_fig.canvas
        network_canvas = network_fig.canvas

        # Render the trend figure onto the combined canvas
        trend_canvas.draw()
        trend_image = np.array(trend_canvas.renderer.buffer_rgba())
        axes[0].imshow(trend_image)
        axes[0].axis("off")

        # Render the network figure onto the combined canvas
        network_canvas.draw()
        network_image = np.array(network_canvas.renderer.buffer_rgba())
        axes[1].imshow(network_image)
        axes[1].axis("off")

        # Save combined figure as a frame
        frame_path = os.path.join(output_folder, f"combined_frame_{t}.png")
        combined_fig.savefig(frame_path, bbox_inches="tight")
        frames.append(imageio.imread(frame_path))

        # Close figures to free memory
        plt.close(trend_fig)
        plt.close(network_fig)
        plt.close(combined_fig)
        os.remove(frame_path)

    # Save all frames as a GIF
    gif_path = os.path.join(output_folder, gif_filename)
    imageio.mimsave(gif_path, frames, duration=0.0005)