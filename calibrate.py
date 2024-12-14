import argparse
from utils.network import *
from utils.measure import *
from utils.nodes import *
from utils.visualization import *
from ast import literal_eval
import sys
import glob
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import CenteredNorm


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
    parser.add_argument("--media_init_mode", type=str, default="gaussian")
    parser.add_argument("--average_media_opinion", type=float, default=0)
    parser.add_argument("--std_media_opinion", type=float, default=0.25)
    parser.add_argument("--extremist_mode_parameter", type=float, default=0.1)
    parser.add_argument("--number_media", type=int, default=40)
    parser.add_argument("--number_media_connection", type=int, default=350)
    parser.add_argument("--media_authority", type=int, default=10)
    parser.add_argument("--threshold_parameter", type=float, default=0.5)
    parser.add_argument("--updated_voters", type=int, default=50)
    parser.add_argument("--initial_threshold", type=list, default=[0, 0.16])
    parser.add_argument("--number_years", type=int, default=5)
    parser.add_argument("--media_feedback_turned_on", type=bool, default=False)
    parser.add_argument("--media_feedback_turned_on_after", type=int, default=10*365)
    parser.add_argument("--media_feedback_probability", type=float, default=0.1)
    parser.add_argument("--media_feedback_threshold_replacement_neutral", type=float, default=0.1)
    parser.add_argument("--number_of_days_election_cycle", type=int, default=50)
    parser.add_argument("--mupdate_parameter_1", type=float, default=2.5)
    parser.add_argument("--mupdate_parameter_2", type=float, default=1)
    # media manipulation parameters
    parser.add_argument("--manipulation_shift", type=float, default=0)
    parser.add_argument("--number_of_manipulated_media", type=int, default=0)
    parser.add_argument("--target_media_opinion", type=float, default=0)
    parser.add_argument("--manipulation_day", type=int, default=4*365)



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
    extr = args.extremist_mode_parameter
    media_mode =args.media_init_mode
    Nm = args.number_media
    Nc = args.number_media_connection
    w = args.media_authority
    alpha = args.threshold_parameter
    Nv = args.updated_voters
    t0 = args.initial_threshold
    Ndays = int(365*args.number_years)
    mfeedback_on = args.media_feedback_turned_on
    mfeedback_after = args.media_feedback_turned_on_after
    number_of_days_election_cycle = args.number_of_days_election_cycle
    mfeedback_prob = args.media_feedback_probability
    mfeedback_threshold_replacement = args.media_feedback_threshold_replacement_neutral
    x = args.mupdate_parameter_1
    y = args.mupdate_parameter_2
    # manipulation parameters
    manipulation_shift = args.manipulation_shift
    number_of_manipulated_media = args.number_of_manipulated_media
    target_media_opinion = args.target_media_opinion
    manipulation_day = args.manipulation_day
    
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
    
    folder = "gaussian"
    print_parameters(args, folder, "parameters.txt")
    network = init_network(df_conx, L, mfeedback_prob, mfeedback_threshold_replacement)  # LxL network of voters
    media = generate_media_landscape(Nm, media_mode, extr=extr, sigma=sigma) 
    media_conx(network, media, Nc)  # Nc random connections per media node

    op_trend = pd.DataFrame()
    prob_to_change = []
    network_polarization = []
    network_std = []
    network_clustering = []
    changed_voters = 0
    election_results = []

    for days in range(Ndays):
        if days >= 365:        
            # have elections
            if days % number_of_days_election_cycle == 0:
                winner = get_election_winner(network)
                election_results.append(winner)
            media=update_media(days, media,election_results, mu, number_of_days_election_cycle, x, y, manipulation_shift=manipulation_shift)
        #turn media feedback on
        if days == mfeedback_after:
            mfeedback = mfeedback_on
        changed_voters += network_update(network, media, Nv, w, t0, alpha, mfeedback)
        network_polarization.append(polarization(network))
        network_std.append(std_opinion(network))
        network_clustering.append(clustering(network))
        sys.stdout.write(f"\rProgress: ({days+1}/{Ndays}) days completed")
        sys.stdout.flush()
        new_row = opinion_share(network)
        new_row.index = [days]
        op_trend = pd.concat([op_trend, new_row])
        if days % (365) == 0:
            prob_to_change.append([days, changed_voters / (np.size(network))])
            changed_voters = 0
        if days == manipulation_day:
            turn_on_media_manipulation_by_opinion_distance(media=media, N=number_of_manipulated_media, target_opinion=target_media_opinion)
            
    # opinion_trend(op_trend, folder, f"opinion_share4_{i}.pdf")
    # op_trend.to_csv(folder + f"/opinion_trend4_{i}.txt", sep="\t", index=False)

    return op_trend.iloc[-1,1], network_std[-1], network_clustering[-1], network_polarization[-1], prob_to_change[-1][1]


def calibrate_parameters(args=None):
    for i in range(3,4):
        # Define ranges for calibration
        param_ranges = {
            "sigma": [0.1,0.2,0.3,0.4,0.5,0.6,0.7],
        }
        results_folder = "gaussian"
        os.makedirs(results_folder, exist_ok=True)
        summary_log = os.path.join(results_folder, f"calibration_log_{i}.txt")
        logs = []

        # Iterate over combinations of parameters
        for _,sigma in enumerate(param_ranges["sigma"]):
                # Update arguments
                args = parse_args()
                args.std_media_opinion = sigma
                

                final_NV, final_std, final_clustering, final_pol, prob_to_change = run_simulation(args)
                            
                logs.append({
                    "std_media_opinion": sigma,
                    "final_opinion": final_NV,
                    "final_std": final_std,
                    "final_clustering": final_clustering,
                    "final_polarization": final_pol,
                    "prob_to_change": prob_to_change
                })

        # Write results to a text file
        with open(summary_log, "w") as f:
            f.write("std_media_opinion,final_opinion,final_std,final_clustering,final_pol,prob_to_change\n")
            for log in logs:
                f.write(
                    f"{log['std_media_opinion']},{log['final_opinion']},{log['final_std']},"
                    f"{log['final_clustering']},{log['final_polarization']},{log['prob_to_change']}\n"
                )

def plot_calibration_heatmap():
    # File paths
    file_path = 'calibrations/regimes_of_the_network/calibration_log_1.txt'
    output_path = 'calibrations/regimes_of_the_network/calibration_result_2.png'

    # Read the file into a DataFrame
    df = pd.read_csv(file_path)

    # Ensure all columns are numeric (skip the first two for plotting axes)
    df['media_authority'] = pd.to_numeric(df['media_authority'], errors='coerce')
    df['initial_threshold'] = pd.to_numeric(df['initial_threshold'], errors='coerce')
    df['final_opinion'] = pd.to_numeric(df['final_opinion'], errors='coerce')
    df['final_std'] = pd.to_numeric(df['final_std'], errors='coerce')
    df['final_clustering'] = pd.to_numeric(df['final_clustering'], errors='coerce')
    df['final_polarization'] = pd.to_numeric(df['final_polarization'], errors='coerce')
    df['prob_to_change'] = pd.to_numeric(df['prob_to_change'], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna()

    # Get unique sorted values for axes
    x_ticks = sorted(df['initial_threshold'].unique())
    y_ticks = sorted(df['media_authority'].unique())

    # Pivot data for imshow
    pivot_data_opinion = df.pivot(index='media_authority', columns='initial_threshold', values='final_opinion')
    pivot_data_std = df.pivot(index='media_authority', columns='initial_threshold', values='final_std')
    pivot_data_clustering = df.pivot(index='media_authority', columns='initial_threshold', values='final_clustering')
    pivot_data_pol = df.pivot(index='media_authority', columns='initial_threshold', values='final_polarization')
    pivot_data_prob_to_change = df.pivot(index='media_authority', columns='initial_threshold', values='prob_to_change')

    midpoints = {
        'final_opinion': 0.172,  # Example midpoint for final_opinion
        'final_std': 0.018,   # Example midpoint for final_std
        'final_clustering': 0.58,  # Example midpoint for final_clustering
        'prob_to_change': 0.95  # Example midpoint for prob_to_change
    }
    norms = {
        'final_opinion': CenteredNorm(
                                    vcenter=midpoints['final_opinion'], 
                                    ),
        'final_std': CenteredNorm( 
                                vcenter=midpoints['final_std'], 
                                ),
        'final_clustering': CenteredNorm( 
                                        vcenter=midpoints['final_clustering'], 
                                       ),
        'prob_to_change': CenteredNorm( 
                                        vcenter=midpoints['prob_to_change'], 
                                        )
    }

    # Plot using imshow
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    spec = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], figure=fig)
    axes = [fig.add_subplot(spec[i, j]) for i in range(2) for j in range(2)]
    plots = [pivot_data_opinion, pivot_data_std, pivot_data_clustering, pivot_data_prob_to_change]
    titles = ['Non-Voters', 'Standard Deviation', 'Clustering', 'Opinion Changes per Year per Voter']
    
    # fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')  # Use a serif font for LaTeX style
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')  # Use sans-serif math mode
    plt.rcParams.update({
        'font.size': 18,  # General font size
        'axes.titlesize': 20,  # Title font size
        'axes.labelsize': 20,  # Label font size
        'legend.fontsize': 16,  # Legend font size
        'xtick.labelsize': 16,  # x-axis tick font size
        'ytick.labelsize': 16   # y-axis tick font size
    })

    for idx, (ax, data, title, key) in enumerate(zip(axes, plots, titles, norms.keys())):
        cax = ax.imshow(data, cmap='seismic', norm=norms[key],  origin='lower', aspect='equal')
        ax.set_title(title)        
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=16)
      # Set ticks to match the pivot table indices and columns
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels([f"{val:.2f}" for val in x_ticks], rotation=45, fontsize=16)
        ax.set_yticks(range(len(y_ticks)))
        ax.set_yticklabels([f"{val:.1f}" for val in y_ticks], fontsize=16)

        # Remove tick labels for middle plots
        if idx % 2 == 1:  # Right column
            ax.set_yticklabels([])
        if idx < 2:  # Top row
            ax.set_xticklabels([])
        if idx % 2 == 0:  # Left column plots
            ax.set_ylabel(r'Media Authority', fontsize=18)
        if idx >= 2:
            ax.set_xlabel(r'$T^0_{NV \rightarrow V}$', fontsize=18)            
    # Save the plot
    plt.savefig(output_path)
    
def plot_calibration_init_threshold():
    # File paths
    file_path = 'calibrations/initial_threshold_calibration_results/calibration_log.txt'
    output_path = 'calibrations/initial_threshold_calibration_results/calibration_result_3.pdf'

    # Read the file into a DataFrame
    df = pd.read_csv(file_path, index_col=False)

    # Ensure all columns are numeric
    df['initial_threshold_1'] = df['initial_threshold_1'].str.replace("(", "").str.replace(")", "")
    df['initial_threshold_2'] = df['initial_threshold_2'].str.replace("(", "").str.replace(")", "")
    df['initial_threshold_1'] = pd.to_numeric(df['initial_threshold_1'], errors='coerce')
    df['initial_threshold_2'] = pd.to_numeric(df['initial_threshold_2'], errors='coerce')
    df['final_opinion'] = pd.to_numeric(df['final_opinion'], errors='coerce')
    df['final_std'] = pd.to_numeric(df['final_std'], errors='coerce')
    df['final_clustering'] = pd.to_numeric(df['final_clustering'], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna()

    # Get unique sorted values for axes
    x_ticks = sorted(df['initial_threshold_2'].unique())
    y_ticks = sorted(df['initial_threshold_1'].unique())

    # Pivot data for imshow
    pivot_data_opinion = df.pivot(index='initial_threshold_1', columns='initial_threshold_2', values='final_opinion')
    pivot_data_std = df.pivot(index='initial_threshold_1', columns='initial_threshold_2', values='final_std')
    pivot_data_clustering = df.pivot(index='initial_threshold_1', columns='initial_threshold_2', values='final_clustering')

    # midpoints = {
    #     'final_opinion': 0.172,  # Example midpoint for final_opinion
    #     'final_std': 0.018,   # Example midpoint for final_std
    #     'final_clustering': 0.58,  # Example midpoint for final_clustering
    #     'prob_to_change': 0.95  # Example midpoint for prob_to_change
    # }
    # norms = {
    #     'final_opinion': CenteredNorm(
    #                                 vcenter=midpoints['final_opinion'], 
    #                                 ),
    #     'final_std': CenteredNorm( 
    #                             vcenter=midpoints['final_std'], 
    #                             ),
    #     'final_clustering': CenteredNorm( 
    #                                     vcenter=midpoints['final_clustering'], 
    #                                    ),
    #     'prob_to_change': CenteredNorm( 
    #                                     vcenter=midpoints['prob_to_change'], 
    #                                     )
    # }

    # Plot using imshow
    fig = plt.figure(figsize=(5, 14), constrained_layout=True)
    spec = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], figure=fig)
    axes = [fig.add_subplot(spec[i, j]) for i in range(3) for j in range(1)]
    plots = [pivot_data_opinion, pivot_data_std, pivot_data_clustering]
    titles = ['Non-Voters', 'Standard Deviation', 'Clustering']
    
    # fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')  # Use a serif font for LaTeX style
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')  # Use sans-serif math mode
    plt.rcParams.update({
        'font.size': 16,  # General font size
        'axes.titlesize': 20,  # Title font size
        'axes.labelsize': 18,  # Label font size
        'legend.fontsize': 16,  # Legend font size
        'xtick.labelsize': 16,  # x-axis tick font size
        'ytick.labelsize': 16   # y-axis tick font size
    })

    for idx, (ax, data, title) in enumerate(zip(axes, plots, titles)):
        cax = ax.imshow(data, cmap='seismic',  origin='lower', aspect='equal')
        ax.set_title(title)        
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
        cbar.ax.tick_params(labelsize=16)
      # Set ticks to match the pivot table indices and columns
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels([f"{val:.2f}" for val in x_ticks], rotation=45, fontsize=14)
        ax.set_yticks(range(len(y_ticks)))
        ax.set_yticklabels([f"{val:.2f}" for val in y_ticks], fontsize=14)
        ax.set_ylabel(r'$T^0_{V \rightarrow NV}$', fontsize=16)
        if idx == 2:
            ax.set_xlabel(r'$T^0_{NV \rightarrow V}$', fontsize=16)            
    # Save the plot
    plt.savefig(output_path)

def plot_calibration_media_feedback():
    # Define file paths and load data from all files
    file_paths = glob.glob("calibrations/mfeedback1_calibration_results/calibration_log*.txt")
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

    # fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')  # Use a serif font for LaTeX style
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')  # Use sans-serif math mode
    plt.rcParams.update({
        'font.size': 18,  # General font size
        'axes.titlesize': 20,  # Title font size
        'axes.labelsize': 20,  # Label font size
        'legend.fontsize': 16,  # Legend font size
        'xtick.labelsize': 16,  # x-axis tick font size
        'ytick.labelsize': 16   # y-axis tick font size
    })
    
    # Plot the results
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    spec = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], figure=fig)
    axs = [fig.add_subplot(spec[i, j]) for i in range(2) for j in range(2)]
    parameters = [
        ("final_opinion", "Non-Voters"),
        ("final_std", "Standard Deviation"),
        ("final_clustering", "Clustering"),
        ("prob_to_change", "Opinion Changes per Year per Voter")
    ]

    for i, (ax, (col_prefix, title)) in enumerate(zip(axs, parameters)):
        ax.errorbar(
            aggregated_data["media_feedback_probability"],
            aggregated_data[f"{col_prefix}_mean"],
            yerr=aggregated_data[f"{col_prefix}_std"],
            fmt='-o',
            capsize=8,
            elinewidth=1.8,
            markersize=10,
            linewidth=1.8,
            label=f"{title} Mean ± Std"
        )
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True)
        ax.legend(loc="upper right")
        if i == 3:
            plt.ylim([0, 0.013])
    axs[-2].set_xlabel("Media Feedback")
    axs[-1].set_xlabel("Media Feedback")
    plt.tight_layout()
    plt.savefig("calibrations/mfeedback1_calibration_results/mfeedback_calibration.pdf")
    plt.savefig("calibrations/mfeedback1_calibration_results/mfeedback_calibration.png")
    
    
def plot_calibration_media_number():
    # Define file paths and load data from all files
    file_paths = glob.glob("calibrations/nmedia_calibration_results/calibration_log*.txt")
    dataframes = [pd.read_csv(file, sep=",") for file in file_paths]

    # Combine all data into a single DataFrame
    combined_data = pd.concat(dataframes, ignore_index=True)
    print(combined_data.columns)

    # Compute the average and standard deviation for each threshold parameter
    aggregated_data = combined_data.groupby("number_media").agg(
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

    # fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')  # Use a serif font for LaTeX style
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')  # Use sans-serif math mode
    plt.rcParams.update({
        'font.size': 18,  # General font size
        'axes.titlesize': 20,  # Title font size
        'axes.labelsize': 20,  # Label font size
        'legend.fontsize': 16,  # Legend font size
        'xtick.labelsize': 16,  # x-axis tick font size
        'ytick.labelsize': 16   # y-axis tick font size
    })
    
    # Plot the results
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    spec = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], figure=fig)
    axs = [fig.add_subplot(spec[i, j]) for i in range(2) for j in range(2)]
    parameters = [
        ("final_opinion", "Non-Voters"),
        ("final_std", "Standard Deviation"),
        ("final_clustering", "Clustering"),
        ("prob_to_change", "Opinion Changes per Year per Voter")
    ]

    for i, (ax, (col_prefix, title)) in enumerate(zip(axs, parameters)):
        ax.errorbar(
            aggregated_data["number_media"],
            aggregated_data[f"{col_prefix}_mean"],
            yerr=aggregated_data[f"{col_prefix}_std"],
            fmt='-o',
            capsize=8,
            elinewidth=1.8,
            markersize=10,
            linewidth=1.8,
            label=f"{title} Mean ± Std"
        )
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True)
        ax.legend(loc="upper right")
        
    axs[-2].set_xlabel("Number Media")
    axs[-1].set_xlabel("Number Media")
    plt.tight_layout()
    plt.savefig("calibrations/nmedia_calibration_results/nmedia_calibration.pdf")
    plt.savefig("calibrations/nmedia_calibration_results/nmedia_calibration.png")
    
    
def plot_calibration_alpha():
    # Define file paths and load data from all files
    file_paths = glob.glob("calibrations/alpha_calibration_results_1/calibration_log*.txt")
    dataframes = [pd.read_csv(file, sep=",") for file in file_paths]

    # Combine all data into a single DataFrame
    combined_data = pd.concat(dataframes, ignore_index=True)
    print(combined_data.columns)

    # Compute the average and standard deviation for each threshold parameter
    aggregated_data = combined_data.groupby("threshold_parameter").agg(
        final_opinion_mean=("final_opinion", "mean"),
        final_opinion_std=("final_opinion", "std"),
        final_std_mean=("final_std", "mean"),
        final_std_std=("final_std", "std"),
        final_clustering_mean=("final_clustering", "mean"),
        final_clustering_std=("final_clustering", "std"),
        final_pol_mean=("final_pol", "mean"),
        final_pol_std=("final_pol", "std"),
        prob_to_change_mean=("final_pol", "mean"),
        prob_to_change_std=("final_pol", "std")
    ).reset_index()

    # fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')  # Use a serif font for LaTeX style
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')  # Use sans-serif math mode
    plt.rcParams.update({
        'font.size': 18,  # General font size
        'axes.titlesize': 20,  # Title font size
        'axes.labelsize': 20,  # Label font size
        'legend.fontsize': 16,  # Legend font size
        'xtick.labelsize': 16,  # x-axis tick font size
        'ytick.labelsize': 16   # y-axis tick font size
    })
    
    # Plot the results
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    spec = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], figure=fig)
    axs = [fig.add_subplot(spec[i, j]) for i in range(2) for j in range(2)]
    parameters = [
        ("final_opinion", "Non-Voters"),
        ("final_std", "Standard Deviation"),
        ("final_clustering", "Clustering"),
        ("final_pol", "Voter Polarization")
    ]

    for i, (ax, (col_prefix, title)) in enumerate(zip(axs, parameters)):
        ax.errorbar(
            aggregated_data["threshold_parameter"],
            aggregated_data[f"{col_prefix}_mean"],
            yerr=aggregated_data[f"{col_prefix}_std"],
            fmt='-o',
            capsize=8,
            elinewidth=1.8,
            markersize=10,
            linewidth=1.8,
            label=f"{title} Mean ± Std"
        )
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True)
        ax.legend(loc="upper right")
        
    axs[-2].set_xlabel("Threshold parameter")
    axs[-1].set_xlabel("Threshold parameter")
    plt.tight_layout()
    plt.savefig("calibrations/alpha_calibration_results_1/alpha_calibration.pdf")
    plt.savefig("calibrations/alpha_calibration_results_1/alpha_calibration.png")
    
    
def plot_extremism():
    # Define file paths and load data from all files
    file_paths = glob.glob("gaussian/calibration_log*.txt")
    dataframes = [pd.read_csv(file, sep=",") for file in file_paths]

    # Combine all data into a single DataFrame
    combined_data = pd.concat(dataframes, ignore_index=True)
    print(combined_data.columns)

    # Compute the average and standard deviation for each threshold parameter
    aggregated_data = combined_data.groupby("std_media_opinion").agg(
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

    # fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')  # Use a serif font for LaTeX style
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')  # Use sans-serif math mode
    plt.rcParams.update({
        'font.size': 18,  # General font size
        'axes.titlesize': 20,  # Title font size
        'axes.labelsize': 20,  # Label font size
        'legend.fontsize': 16,  # Legend font size
        'xtick.labelsize': 16,  # x-axis tick font size
        'ytick.labelsize': 16   # y-axis tick font size
    })
    
    # Plot the results
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    spec = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], figure=fig)
    axs = [fig.add_subplot(spec[i, j]) for i in range(2) for j in range(2)]
    parameters = [
        ("final_opinion", "Non-Voters"),
        ("final_std", "Standard Deviation"),
        ("final_clustering", "Clustering"),
        ("prob_to_change", "Opinion Changes per Year per Voter")
    ]

    for ax, (col_prefix, title) in zip(axs, parameters):
        ax.errorbar(
            aggregated_data["std_media_opinion"],
            aggregated_data[f"{col_prefix}_mean"],
            yerr=aggregated_data[f"{col_prefix}_std"],
            fmt='-o',
            capsize=8,
            elinewidth=1.8,
            markersize=10,
            linewidth=1.8,
            label=f"{title} Mean ± Std"
        )
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True)
        ax.legend(loc="upper right")
    axs[-2].set_xlabel(r"$\sigma$")
    axs[-1].set_xlabel(r"$\sigma$")
    plt.tight_layout()
    plt.savefig("gaussian/results.pdf")
    plt.savefig("gaussian/results.png")


if __name__ == "__main__":
    _args = parse_args()
    # calibrate_parameters(_args)
    # plot_calibration_media_feedback()
    # plot_calibration_media_number()
    plot_calibration_init_threshold()
    # plot_calibration_alpha()
    # plot_calibration_heatmap()
    # plot_extremism()
