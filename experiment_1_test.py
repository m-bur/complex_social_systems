import multiprocessing
import subprocess
import numpy as np

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
    folder_name = "Run_1"
    parameters = []
    number_of_media = np.arange(1,25,3)
    media_shift = 0.8 / number_of_media
    target_opinion = np.linspace(-1, 0.9, num=12)
    average_factor = 8
    for m_shift, n_media in zip(media_shift, number_of_media):
        for t_opinion in target_opinion:
            for i in range(average_factor):
                name = folder_name + f"_n_{n_media}" + f"_o_{round(t_opinion,3)}".replace(".","p")
                parameters.append(["--number_of_manipulated_media", str(n_media), "--manipulation_shift", str(m_shift), "--target_media_opinion", str(t_opinion), "--parent_folder", name, "number_years", "5"])
    
    return parameters
# Example usage
if __name__ == "__main__":
    # List of parameter sets to pass to main.py
    parameters = generate_parameters()

    # Define the maximum number of cores to use
    MAX_CORES = 1000
    # Run in parallel
    run_in_parallel(parameters, MAX_CORES)
