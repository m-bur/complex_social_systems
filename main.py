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
    parser.add_argument("--media_init_mode", type=str, default="fixed")
    parser.add_argument("--average_media_opinion", type=float, default=0)
    parser.add_argument("--std_media_opinion", type=float, default=0.25)
    parser.add_argument("--number_media", type=int, default=40)
    parser.add_argument("--number_media_connection", type=int, default=500)
    parser.add_argument("--media_authority", type=int, default=10)
    parser.add_argument("--threshold_parameter", type=float, default=0.5)
    parser.add_argument("--updated_voters", type=int, default=50)
    parser.add_argument("--initial_threshold", type=list, default=[0, 0.16])
    parser.add_argument("--number_years", type=int, default=2)
    parser.add_argument("--media_feedback_turned_on", type=bool, default=False)
    parser.add_argument("--number_of_days_election_cycle", type=int, default=5)
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
    media_mode =args.media_init_mode
    Nm = args.number_media
    Nc = args.number_media_connection
    w = args.media_authority
    alpha = args.threshold_parameter
    Nv = args.updated_voters
    t0 = args.initial_threshold
    Ndays = 500#int((365*1.5)//1) #args.number_years
    initial_media_opinion=0#media value to start with
    mfeedback_on = True#args.media_feedback_turned_on
    number_of_days_election_cycle = 10#args.number_of_days_election_cycle

    if regen_network:
        df_conx = init_df_conx(c_min, c_max, gamma, L)
        first_conx_matrix = calc_first_conx_matrix(L, L_G)
        second_conx_matrix = calc_second_conx_matrix(L, first_conx_matrix, p_c)
        connection_matrix = first_conx_matrix + second_conx_matrix
        df_conx = update_df_conx(L, df_conx, connection_matrix)
        df_conx.to_csv(network_path)
    else:
        df_conx = pd.read_csv(network_path, converters={"connection": literal_eval})

    folder = make_foldername(base_name="Figure_collection/figures")
    print_parameters(args, folder, "parameters.txt")
    network = init_network(df_conx, [[Voter(i, j) for i in range(L)] for j in range(L)])  # LxL network of voters
    deg_distribution(network, folder, "deg_distribution.pdf")
    media = generate_media_landscape(Nm, media_mode)
    media_conx(network, media, Nc)  # Nc random connections per media node
    number_media_distribution(network, folder, "number_media_distribution.pdf")
    neighbor_opinion_distribution(network, folder, "initial_neighbour_dist.pdf")
    visualize_network(network, folder, "initial_network.pdf")

    op_trend = pd.DataFrame()
    prob_to_change = []
    network_polarization = []
    network_std = []
    network_clustering = []
    changed_voters = 0

    election_results = []

    for days in range(Ndays):
        #was mues im loop sie: media.set
        #active this to update media opinion:
        #media=update_media(days,media,election_results, initial_media_opinion, number_of_days_election_cycle, media_update_cycle=4 )
        initial_media_opinion=0#media change only at start != zero

        changed_voters += network_update(network, media, Nv, w, t0, alpha, mfeedback_on)
        op_trend = pd.concat([op_trend, opinion_share(network)], ignore_index=True)
        network_polarization.append(polarization(network))
        network_std.append(std_opinion(network))
        network_clustering.append(clustering(network))

        # have elections
        if days % number_of_days_election_cycle == 0:
            winner = get_election_winner(network)
            election_results.append(winner)

        # progress bar #####################
        sys.stdout.write(f"\rProgress: ({days+1}/{Ndays}) days completed\n")
        sys.stdout.flush()
        # progress bar #####################
        if days % (4*365) == 0:
            prob_to_change.append([days, changed_voters / (4 * np.size(network))])
        if days == 1000:
            mfeedback_on = True
    opinion_trend(op_trend, folder, "opinion_share.pdf")
    op_trend.to_csv(folder + "/opinion_trend.txt", sep="\t", index=False)
    plot_polarizaiton(network_polarization, folder, "network_polarization.pdf")
    print_measure(network_polarization, folder, "network_polarizaiton.txt")
    plot_std(network_std, folder, "network_std.pdf")
    print_measure(network_std, folder, "network_std.txt")
    plot_prob_to_change(prob_to_change, folder, "prob_to_change.pdf")
    print_prob_to_change(prob_to_change, folder, "prob_to_change.txt")
    plot_clustering(network_clustering, folder, "network_clustering.pdf")
    print_measure(network_clustering, folder, "network_clustering.txt")
    neighbor_opinion_distribution(network, folder, "final_neighbour_dist.pdf")
    visualize_network(network, folder, "final_network.pdf")


if __name__ == "__main__":
    _args = parse_args()
    main(_args)
