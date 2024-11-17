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
    parser.add_argument("--number_media", type=int, default=100)
    parser.add_argument("--number_media_connection", type=int, default=350)
    parser.add_argument("--media_authority", type=int, default=10)
    parser.add_argument("--threshold_parameter", type=float, default=0.5)
    parser.add_argument("--updated_voters", type=int, default=50)
    parser.add_argument("--initial_threshold", type=list, default=[0, 0.14])
    parser.add_argument("--number_years", type=int, default=5)
    parser.add_argument("--media_feedback_turned_on", type=bool, default=False)
    return parser.parse_args()


def run_simulation(args):
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
    media = generate_media_landscape(Nm, media_mode) 
    media_conx(network, media, Nc)  # Nc random connections per media node

    op_trend = pd.DataFrame()
    prob_to_change = []
    network_polarization = []
    network_std = []
    network_clustering = []
    changed_voters = 0

    for days in range(Ndays):
        changed_voters += network_update(network, media, Nv, w, t0, alpha, mfeedback_on)
        op_trend = pd.concat([op_trend, opinion_share(network)], ignore_index=True)
        network_polarization.append(polarization(network))
        network_std.append(std_opinion(network))
        network_clustering.append(clustering(network))
        sys.stdout.write(f"\rProgress: ({days+1}/{Ndays}) days completed")
        sys.stdout.flush()
        if days % (4*365) == 0:
            prob_to_change.append([days, changed_voters / (4 * np.size(network))])

    return op_trend.iloc[-1,1], network_std[-1], network_clustering[-1]

def calibrate_parameters(args=None):
    # Define ranges for calibration
    param_ranges = {
        "initial_threshold": [(0, 0.12), (0, 0.14), (0, 0.16), (0, 0.18), (0, 0.2), (0.05, 0.12), (0.05, 0.14), (0.05, 0.16), (0.05, 0.18), (0.05, 0.2), (0.1, 0.12), (0.1, 0.14), (0.1, 0.16), (0.1, 0.18), (0.1, 0.2), (0.15, 0.12), (0.15, 0.14), (0.15, 0.16), (0.15, 0.18), (0.15, 0.2), (0.2, 0.12), (0.2, 0.14), (0.2, 0.16), (0.2, 0.18), (0.2, 0.2)],  # Pairs of thresholds to test
    }
    results_folder = "initial_threshold_calibration_results"
    os.makedirs(results_folder, exist_ok=True)
    summary_log = os.path.join(results_folder, "calibration_log.txt")
    logs = []

    # Iterate over combinations of parameters
    for initial_threshold in param_ranges["initial_threshold"]:
        # Update arguments
        args = parse_args()
        args.initial_threshold = initial_threshold

        final_NV, final_std, final_clustering = run_simulation(args)
                        
        logs.append({
            "initial_threshold": initial_threshold,
            "final_opinion": final_NV,
            "final_std": final_std,
            "final_clustering": final_clustering,
        })

    # Write results to a text file
    with open(summary_log, "w") as f:
        f.write("Calibration Results\n")
        f.write("initial_threshold,final_opinion,final_std,final_clustering,status\n")
        for log in logs:
            f.write(
                f"{log['initial_threshold']},{log['final_opinion']},{log['final_std']},"
                f"{log['final_clustering']}\n"
            )


if __name__ == "__main__":
    _args = parse_args()
    calibrate_parameters(_args)
