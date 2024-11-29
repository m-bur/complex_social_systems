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
    num_processes = min(max_cores, multiprocessing.cpu_count())
    
    # Create a pool of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(run_main_with_params, params_list)

# Example usage
if __name__ == "__main__":
    # List of parameter sets to pass to main.py
    parameters = [
        ["--parent_folder", "Folder_A", "--number_years", "0.1"],
        ["--parent_folder", "Folder_B", "--number_years", "0.2"],
        ["--parent_folder", "Folder_C", "--number_years", "0.1"],
        ["--parent_folder", "Folder_D", "--number_years", "0.2"],
        ["--parent_folder", "Folder_A", "--number_years", "0.1"],
        ["--parent_folder", "Folder_B", "--number_years", "0.2"],
        ["--parent_folder", "Folder_C", "--number_years", "0.1"],
        ["--parent_folder", "Folder_D", "--number_years", "0.2"]
    ]

    # Define the maximum number of cores to use
    MAX_CORES = 4

    # Run in parallel
    run_in_parallel(parameters, MAX_CORES)
