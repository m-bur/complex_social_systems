import random
import numpy as np
import pandas as pd


def first_connection_p(l, L_G, L):
    a = L_G
    b = a / 4
    return 1 / (1 + np.exp((l - a) / b)) + 0.001 * (L - l) / L


def build_first_connection(L_G=20, L=100):
    df_first_con = pd.DataFrame(columns=["x", "y", "num", "first_connection"])
    # Loop over the L * L network to create each node
    for i in range(L):
        for j in range(L):

            neighbor_list = []  # Initialize an empty list
            # For each node, loop over the network again to create the first connection
            for m in range(L):
                for n in range(L):

                    # Determine connections based on probability after the node
                    if m > i or (m == i and n > j):
                        l = np.sqrt((i - m) ** 2 + (j - n) ** 2)
                        r = random.random()
                        p = first_connection_p(l, L_G, L)
                        if r < p:
                            neighbor_list.append((m, n))

                    # Search the existing connection before the node
                    elif m < i or (m == i and n < j):
                        idx = L * m + n
                        search_list = df_first_con.at[idx, "first_connection"]
                        if (i, j) in search_list:
                            neighbor_list.append((m, n))
                    else:
                        pass

            # Store the row with 'x', 'y', 'number connections', and 'connection list'
            df_first_con.loc[len(df_first_con)] = [
                i,
                j,
                len(neighbor_list),
                neighbor_list,
            ]

        print(i)
