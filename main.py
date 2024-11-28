import argparse
from utils.network import *
from utils.measure import *
from utils.nodes import *
from utils.visualization import *
from ast import literal_eval
import sys
import copy



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
    parser.add_argument("--number_media_connection", type=int, default=350)
    parser.add_argument("--media_authority", type=int, default=10)
    parser.add_argument("--threshold_parameter", type=float, default=0.5)
    parser.add_argument("--updated_voters", type=int, default=50)
    parser.add_argument("--initial_threshold", type=list, default=[0, 0.16])
    parser.add_argument("--number_years", type=float, default=100)
    parser.add_argument("--media_feedback_turned_on", type=bool, default=True)
    parser.add_argument("--media_feedback_probability", type=float, default=0.1)
    parser.add_argument("--media_feedback_threshold_replacement_neutral", type=float, default=0.1)
    parser.add_argument("--number_of_days_election_cycle", type=int, default=50)
    parser.add_argument("--mupdate_parameter_1", type=float, default=2.5)
    parser.add_argument("--mupdate_parameter_2", type=float, default=1)
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
    Ndays = int(365*args.number_years)
    mfeedback_on = args.media_feedback_turned_on
    number_of_days_election_cycle = args.number_of_days_election_cycle
    mfeedback_prob = args.media_feedback_probability
    mfeedback_threshold_replacement = args.media_feedback_threshold_replacement_neutral
    x = args.mupdate_parameter_1
    y = args.mupdate_parameter_2
    
    mfeedback=False

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
    network = init_network(df_conx, L, mfeedback_prob, mfeedback_threshold_replacement)  # LxL network of voters
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
    networks = []
    changed_voters = 0
    media_stats = pd.DataFrame()

    election_results = []
   

    for days in range(Ndays):
        
        # start with elections after the first year
        if days >= 365:        
            # have elections
            if days % number_of_days_election_cycle == 0:
                winner = get_election_winner(network)
                election_results.append(winner)
            media=update_media(days, media,election_results, mu, number_of_days_election_cycle, x, y, manipulation_shift = 0.5)

        changed_voters += network_update(network, media, Nv, w, t0, alpha, mfeedback)

        # measure the network characteristics
        network_polarization.append(polarization(network))
        network_std.append(std_opinion(network))
        network_clustering.append(clustering(network))
        media_stats = pd.concat([media_stats, media_statistics(media=media)], ignore_index=True)

        # progress bar #####################
        sys.stdout.write(f"\rProgress: ({days+1}/{Ndays}) days completed")
        sys.stdout.flush()
        
        # update the changed voters once per year
        if days % (365) == 0:
            prob_to_change.append([days, changed_voters / (np.size(network))])
            changed_voters = 0
        
        #every 5th day, for gif visualization
        if days % 5 == 0:
            #networks.append(copy.deepcopy(network))
            new_row = opinion_share(network)
            new_row.index = [days]
            op_trend = pd.concat([op_trend, new_row])

        #turn media feedback on
        if days == 10*365:
            mfeedback = mfeedback_on

        if days == 700:
            turn_on_media_manipulation_by_opinion_distance(media=media, N=4, target_opinion=-1)
    # plot and save the network charactersitics 
            
    #combined_visualization(op_trend, networks, folder)
    opinion_trend(op_trend, folder, "opinion_share.pdf")
    op_trend.to_csv(folder + "/opinion_trend.txt", sep="\t", index=False)
    plot_polarization(network_polarization, folder, "network_polarization.pdf")
    print_measure(network_polarization, folder, "network_polarizaiton.txt")
    plot_std(network_std, folder, "network_std.pdf")
    print_measure(network_std, folder, "network_std.txt")
    plot_prob_to_change(prob_to_change, folder, "prob_to_change.pdf")
    print_prob_to_change(prob_to_change, folder, "prob_to_change.txt")
    plot_clustering(network_clustering, folder, "network_clustering.pdf")
    print_measure(network_clustering, folder, "network_clustering.txt")
    neighbor_opinion_distribution(network, folder, "final_neighbour_dist.pdf")
    visualize_network(network, folder, "final_network.pdf")
    print_media_statistics(df_stats=media_stats, output_folder=folder)
    plot_media_stats(df_stats=media_stats, output_folder=folder)
    plot_media_shares(df_stats=media_stats, output_folder=folder)
    df_consecutive_terms = get_consecutive_terms_counts(election_results=election_results)
    plot_consecutive_terms_histogram(df_consecutive_terms, output_folder=folder, file_name="consecutive_terms.pdf")
    print_election_results(election_results, folder=folder, filename="election_results.txt")


if __name__ == "__main__":
    _args = parse_args()
    main(_args)
