import multiprocessing
import subprocess


def run_main_with_params(params):
    """
    Runs the main.py file with the given parameters on a separate process.

    :param params: A list of arguments to pass to main.py
    """
    try:
        # Construct the command
        command = ["python", "main.py"] + params

        # Run the command and capture the output
        result = subprocess.run(command, text=True, capture_output=True)

        print(f"Running main.py with params: {params}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")

    except Exception as e:
        print(f"An error occurred while running main.py with params {params}: {e}")


def run_in_parallel(params_list, max_cores):
    """
    Runs each set of parameters in parallel, limited to max_cores processes.

    :param params_list: A list of parameter sets (each set is a list of arguments)
    :param max_cores: Maximum number of cores to use for parallel processing
    """
    # Use the minimum of max_cores and available CPU cores
    num_processes = min(max_cores, multiprocessing.cpu_count()) - 2

    # Create a pool of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_main_with_params, params_list)


def generate_parameters():
    folder_name = "Init_media_distributions"
    parameters = []
    media_dists = ["extremist"]
    extr = [0.1, 0.5]
    average_factor = 1
    for m_dist in media_dists:
        for e in extr:
            for _ in range(average_factor):
                name = folder_name + f"_{m_dist}" + f"_{e}".replace(".", "p")
                parameters.append(
                    [
                        "--media_init_mode",
                        str(m_dist),
                        "--parent_folder",
                        name,
                        "--number_years",
                        "100",
                        "--extremist_mode_parameter",
                        str(e),
                    ]
                )

    return parameters


# Example usage
if __name__ == "__main__":
    # List of parameter sets to pass to main.py
    parameters = generate_parameters()
    # print(parameters)
    # Define the maximum number of cores to use
    MAX_CORES = 1000
    # Run in parallel
    run_in_parallel(parameters, MAX_CORES)
