import numpy as np


def first_level_connection(l, L_G=20, L=100):
    a = L_G
    b = a / 4
    return 1 / (1 + np.exp((l - a) / b)) + 0.001 * (L - 1) / L

