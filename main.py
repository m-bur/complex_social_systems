
import argparse
from utils import utils
from network import *
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side_length", type=int, default=100)
    parser.add_argument("--local_length", type=int, default=20)
    parser.add_argument("--min_neighbors", type=int, default=18)
    parser.add_argument("--max_neighbors", type=int, default=52)
    parser.add_argument("--prob_first_conx", type=float, default=3.0)
    parser.add_argument("--prob_second_conx", type=float, default=0.1)
    parser.add_argument("--regen_network", type=bool, default=False)
    parser.add_argument("--network_path", type=str, default='network.csv')
    return parser.parse_args()


def main(args=None):
    L = args.side_length
    L_G = args.local_length
    c_min = args.min_neighbors
    c_max = args.max_neighbors
    gamma = args.prob_first_conx
    p_c = args.prob_second_conx
    regen_network = args.regen_network
    network_path = args.network_path

    if regen_network:
        df_conx = init_df_conx(c_min, c_max, gamma, L)
        first_conx_matrix = calc_first_conx_matrix(L, L_G)
        second_conx_matrix = calc_second_conx_matrix(L, first_conx_matrix, p_c)
        connection_matrix = first_conx_matrix + second_conx_matrix
        df_conx = update_df_conx(L, df_conx, connection_matrix)
        df_conx.to_csv(network_path)
    else:
        df_conx = pd.read_csv(network_path)
        dict_voter = {}
        for i in range(len(df_conx)):
            dict_voter[i] = utils.Voter(df_conx.loc[i, 'x'], df_conx.loc[i, 'y'])
            dict_voter[i].set_neighbor(df_conx.loc[i, 'connection'])

    plt.hist(df_conx['num'])
    plt.show()


if __name__ == "__main__":
    _args = parse_args()
    main(_args)
