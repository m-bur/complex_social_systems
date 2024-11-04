print(1)
import numpy as np
from utils.utils import Voter
from utils.degree_distribution import deg_distribution




# Example: Creating a grid of voters and assigning random neighbors

# Parameters
num_rows, num_cols = 10, 10  # Grid size

# Initialize a grid of Voter objects
voters_grid = [[Voter(i, j) for j in range(num_cols)] for i in range(num_rows)]

# Assign random neighbors to each voter
for i in range(num_rows):
    for j in range(num_cols):
        voter = voters_grid[i][j]
        
        # Randomly select up to 4 neighbors
        for _ in range(np.random.randint(1, 10)):
            # Randomly choose neighbor coordinates within grid bounds
            neighbor_i = np.random.randint(0, num_rows)
            neighbor_j = np.random.randint(0, num_cols)
            
            # Avoid self-loops
            if (neighbor_i, neighbor_j) != (i, j):
                voter.add_neighbor(neighbor_i, neighbor_j)

# Example of accessing a voter's data
sample_voter = voters_grid[0][0]
print(sample_voter)
print("Number of neighbors:", sample_voter.get_number_of_neighbors())

deg_distribution(voters_grid)

