import argparse
from utils.network import *
from utils.measure import *
from utils.nodes import *
from utils.visualization import *
from ast import literal_eval
import sys
import glob


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
    parser.add_argument("--number_years", type=int, default=5)
    parser.add_argument("--media_feedback_turned_on", type=bool, default=False)
    parser.add_argument("--media_feedback_probability", type=float, default=0.1)
    parser.add_argument("--media_feedback_threshold_replacement_neutral", type=float, default=0.1)
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
    mfeedback_prob = args.media_feedback_probability
    mfeedback_threshold_replacement = args.media_feedback_threshold_replacement_neutral

    if regen_network:
        df_conx = init_df_conx(c_min, c_max, gamma, L)
        first_conx_matrix = calc_first_conx_matrix(L, L_G)
        second_conx_matrix = calc_second_conx_matrix(L, first_conx_matrix, p_c)
        connection_matrix = first_conx_matrix + second_conx_matrix
        df_conx = update_df_conx(L, df_conx, connection_matrix)
        df_conx.to_csv(network_path)
    else:
        df_conx = pd.read_csv(network_path, converters={"connection": literal_eval})
    
    print_parameters(args, "mfeedback_calibration_results", "parameters.txt")
    network = init_network(df_conx, L, mfeedback_prob, mfeedback_threshold_replacement)  # LxL network of voters
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
        if days % (365) == 0:
            prob_to_change.append([days, changed_voters / (np.size(network))])
            changed_voters = 0
        if days == 2*365:
            mfeedback_on = True

    return op_trend.iloc[-1,1], network_std[-1], network_clustering[-1], network_polarization[-1], prob_to_change[-1][1]


def calibrate_parameters(args=None):
    for i in range(1,6):
        # Define ranges for calibration
        param_ranges = {
            "mfeedback_prob": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        }
        results_folder = "mfeedback_calibration_results"
        os.makedirs(results_folder, exist_ok=True)
        summary_log = os.path.join(results_folder, f"calibration_log_{i}.txt")
        logs = []

        # Iterate over combinations of parameters
        for mfeedback_prob in param_ranges["mfeedback_prob"]:
            # Update arguments
            args = parse_args()
            args.media_feedback_probability = mfeedback_prob

            final_NV, final_std, final_clustering, final_pol, prob_to_change = run_simulation(args)
                            
            logs.append({
                "media_feedback_probability": mfeedback_prob,
                "final_opinion": final_NV,
                "final_std": final_std,
                "final_clustering": final_clustering,
                "final_polarization": final_pol,
                "prob_to_change": prob_to_change
            })

        # Write results to a text file
        with open(summary_log, "w") as f:
            f.write("media_feedback_probability,final_opinion,final_std,final_clustering,final_pol,prob_to_change\n")
            for log in logs:
                f.write(
                    f"{log['media_feedback_probability']},{log['final_opinion']},{log['final_std']},"
                    f"{log['final_clustering']},{log['final_polarization']},{log['prob_to_change']}\n"
                )

def plot_calibration_heatmap():
    # Read the data from the file
    file_path = 'nmedia_calibration_results/calibration_log.txt'
    # Read the file line by line, skipping the header
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        header = lines[0].strip().split(',')  # Extract the header
        for line in lines[1:]:
            row = line.strip().split(',')
            data.append(row)

    # Create a DataFrame manually
    df = pd.DataFrame(data, columns=['x', 'y', 'final_opinion','final_std','final_clustering', 'final_polarization'])

    # Parse the `initial_threshold` column into `x` and `y` coordinates
    df['x'] = df['x'].str.strip('()').astype(float)
    df['y'] = df['y'].str.strip('()').astype(float)

    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['final_opinion'] = pd.to_numeric(df['final_opinion'], errors='coerce')
    df['final_std'] = pd.to_numeric(df['final_std'], errors='coerce')
    df['final_clustering'] = pd.to_numeric(df['final_clustering'], errors='coerce')
    df['final_polarization'] = pd.to_numeric(df['final_polarization'], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna()

    x_ticks = sorted(df['x'].unique())
    y_ticks = sorted(df['y'].unique())

    # Pivot data for imshow
    pivot_data_opinion = df.pivot(index='y', columns='x', values='final_opinion')
    pivot_data_std = df.pivot(index='y', columns='x', values='final_std')
    pivot_data_clustering = df.pivot(index='y', columns='x', values='final_clustering')
    pivot_data_pol = df.pivot(index='y', columns='x', values='final_polarization')

    # Plot using imshow
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), constrained_layout=True)
    plots = [pivot_data_opinion, pivot_data_std, pivot_data_clustering]
    titles = ['Non-Voters', 'Final Std', 'Final Clustering', 'Final Polarization']

    for ax, data, title in zip(axes, plots, titles):
        cax = ax.imshow(data, cmap='viridis', origin='lower', aspect='auto')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('$T_{V\\rightarrow NV}$', fontsize=12)
        ax.set_ylabel('$T_{NV\\rightarrow V}$', fontsize=12)
        fig.colorbar(cax, ax=ax, orientation='vertical')
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels([f"{val:.2f}" for val in x_ticks])
        ax.set_yticks(range(len(y_ticks)))
        ax.set_yticklabels([f"{val:.2f}" for val in y_ticks])

    plt.savefig("initial_threshold_calibration_results/initial_threshold_calibration.pdf")


def plot_calibration():
    # Define file paths and load data from all files
    file_paths = glob.glob("mfeedback_calibration_results/calibration_log*.txt")
    dataframes = [pd.read_csv(file, sep=",") for file in file_paths]

    # Combine all data into a single DataFrame
    combined_data = pd.concat(dataframes, ignore_index=True)
    print(combined_data.columns)

    # Compute the average and standard deviation for each threshold parameter
    aggregated_data = combined_data.groupby("media_feedback_probability").agg(
        final_opinion_mean=("final_opinion", "mean"),
        final_opinion_std=("final_opinion", "std"),
        final_std_mean=("final_std", "mean"),
        final_std_std=("final_std", "std"),
        final_clustering_mean=("final_clustering", "mean"),
        final_clustering_std=("final_clustering", "std"),
        final_pol_mean=("final_pol", "mean"),
        final_pol_std=("final_pol", "std"),
        prob_to_change_mean=("prob_to_change", "mean"),
        prob_to_change_std=("prob_to_change", "std")
    ).reset_index()
    print(aggregated_data)

    # Plot the results
    fig, axs = plt.subplots(5, 1, figsize=(10, 16), sharex=True)
    parameters = [
        ("final_opinion", "Non-Voters"),
        ("final_std", "Final Std"),
        ("final_clustering", "Final Clustering"),
        ("final_pol", "Final Polarization"),
        ("prob_to_change", "Changes in Opinion per year")
    ]

    for ax, (col_prefix, title) in zip(axs, parameters):
        ax.errorbar(
            aggregated_data["media_feedback_probability"],
            aggregated_data[f"{col_prefix}_mean"],
            yerr=aggregated_data[f"{col_prefix}_std"],
            fmt='-o',
            capsize=5,
            label=f"{title} Mean ± Std"
        )
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel("Media Feedback")
    plt.tight_layout()
    plt.savefig("mfeedback_calibration_results/mfeedback_calibration.pdf")

if __name__ == "__main__":
    _args = parse_args()
    #calibrate_parameters(_args)
    plot_calibration()
