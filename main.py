import argparse
from utils.network import *
from utils.measure import *
from utils.nodes import *
from utils.visualization import *
from ast import literal_eval
import sys



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side_length", type=int, default=50)
    parser.add_argument("--local_length", type=int, default=10)
    parser.add_argument("--min_neighbors", type=int, default=18)
    parser.add_argument("--max_neighbors", type=int, default=52)
    parser.add_argument("--prob_first_conx", type=float, default=3.0)
    parser.add_argument("--prob_second_conx", type=float, default=0.2)
    parser.add_argument("--regen_network", type=bool, default=False)
    parser.add_argument("--network_path", type=str, default="network.csv")
    parser.add_argument("--average_media_opinion", type=float, default=0)
    parser.add_argument("--std_media_opinion", type=float, default=1)
    parser.add_argument("--number_media", type=int, default=50)
    parser.add_argument("--number_media_connection", type=int, default=700)
    parser.add_argument("--media_authority", type=int, default=1)
    parser.add_argument("--threshold_parameter", type=float, default=0.7)
    parser.add_argument("--updated_voters", type=int, default=25)
    parser.add_argument("--initial_threshold", type=list, default=[0, 0.18])
    parser.add_argument("--number_years", type=int, default=1)
    parser.add_argument("--media_feedback_turned_on", type=bool, default=False)
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
    mu = args.average_media_opinion
    sigma = args.std_media_opinion
    Nm = args.number_media
    Nc = args.number_media_connection
    w = args.media_authority
    alpha = args.threshold_parameter
    Nv = args.updated_voters
    t0 = args.initial_threshold
    Ndays = 365*args.number_years
    mfeedback_on = args.media_feedback_turned_on

    if regen_network:
        df_conx = init_df_conx(c_min, c_max, gamma, L)
        first_conx_matrix = calc_first_conx_matrix(L, L_G)
        second_conx_matrix = calc_second_conx_matrix(L, first_conx_matrix, p_c)
        connection_matrix = first_conx_matrix + second_conx_matrix
        df_conx = update_df_conx(L, df_conx, connection_matrix)
        df_conx.to_csv(network_path)
    else:
        df_conx = pd.read_csv(network_path, converters={"connection": literal_eval})

    network = init_network(df_conx, [[Voter(i, j) for i in range(L)] for j in range(L)])  # LxL network of voters
    deg_distribution(network)
    media = generate_media_landscape(Nm, "fixed")  # Nm media network with average opinion mu
    media_conx(network, media, Nc)  # Nc random connections per media node
    number_media_distribution(network)

    neighbor_opinion_distribution(network, "initial_neighbour_dist")
    visualize_network(network, "initial_network.png")

    op_trend = pd.DataFrame()
    prob_to_change = []
    network_polarization = []
    network_std = []
    changed_voters = 0

    for days in range(Ndays):
        changed_voters += network_update(network, media, Nv, w, t0, alpha, mfeedback_on)
        op_trend = pd.concat([op_trend, opinion_share(network)], ignore_index=True)
        network_polarization.append(polarization(network))
        network_std.append(std_opinion(network))
        sys.stdout.write(f"\rProgress: ({days+1}/{Ndays}) days completed")
        sys.stdout.flush()
        if days // 365 == 4:
            prob_to_change.append(changed_voters / (4 * np.size(network)))

    opinion_trend(op_trend)
    plot_polarizaiton(network_polarization)
    plot_std(network_std)
    plot_prob_to_change(prob_to_change)
    neighbor_opinion_distribution(network, "final_neighbour_dist")
    visualize_network(network, "final_network.png")


if __name__ == "__main__":
    _args = parse_args()
    main(_args)
