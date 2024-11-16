import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from matplotlib.colors import ListedColormap, BoundaryNorm


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


def test_visualize_matrix():
    """
    Test the `visualize_matrix` function by creating and visualizing a sample matrix.

    This function generates a 1000x1000 matrix with specific structured regions:
    a red background, a blue circle in the center, and a grey ring between the circle
    and background. It then calls `visualize_matrix` to create and save the image with
    a unique filename. The test checks if the image was saved correctly.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Create a 1000x1000 matrix with default grey (0)
    size = 1000
    matrix = np.zeros((size, size), dtype=int)

    # Create a red background (-1)
    matrix[:, :] = -1

    # Create a blue circle (1) in the center
    center = size // 2
    radius = size // 3
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
    mask_circle = dist_from_center <= radius
    matrix[mask_circle] = 1  # Blue circle

    # Create a grey ring (0) between blue circle and red background
    inner_radius = size // 3
    outer_radius = size // 2
    mask_ring = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
    matrix[mask_ring] = 0  # Grey ring

    # Generate a unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"matrix_visualization_{timestamp}.png"

    # Call the visualize_matrix function
    output_folder = "test_output"
    visualize_matrix(matrix, output_folder, filename=filename)

    # Verify if the image file was created
    output_path = os.path.join(output_folder, filename)
    if os.path.isfile(output_path):
        print(f"Test passed: Image successfully saved to {output_path}")
    else:
        print(f"Test failed: Image not found at {output_path}")

# Run the test function
#test_visualize_matrix()

def visualize_network(network, filename=None):
    """visualize voter network"""
    n = np.shape(network)
    matrix = np.zeros((n[0], n[1]), dtype=int)
    for i in range(n[0]):
        for j in range(n[1]):
            matrix[i, j] = network[i][j].get_opinion()
    visualize_matrix(matrix, "figures", filename)