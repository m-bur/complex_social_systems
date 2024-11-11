from utils import build_first_connection


L = 50       # side length
L_G = 10        # neighbor side length
a = L_G         # parameter for probability function
b = a / 4       # parameter for probability function
N = L ** 2      # total number of nodes


df_first_con = build_first_connection(L_G, L)

