
from utils import first_level_connection


L = 100         # side length
L_G = 20        # neighbor side length
a = L_G         # parameter for probability function
b = a / 4       # parameter for probability function
N = L ** 2      # total number of nodes


val_try = first_level_connection(30)
