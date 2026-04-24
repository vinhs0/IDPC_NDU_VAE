import os
import time
import statistics
import torch
import numpy as np

# Import configs and task definition
from vae.VAEConfigs import Configs
from problem.IDPCNDU import IDPCNDU

# Import your newly created offlineQD class
from vae.offlineQD import offlineQD 

def run_offline_tasks(fw, tasks_info):
    print(f"Starting Offline Execution for {len(tasks_info)} tasks...")

    # Dictionaries to store statistics
    bf = {i: float('inf') for i in range(len(tasks_info))}
    rs = {i: [] for i in range(len(tasks_info))}
    time_records = {i: [] for i in range(len(tasks_info))}

    for i, (d_path, o_dir, name) in enumerate(tasks_info):
        print(f"\n=========================================")
        print(f"Processing Task: {name}")
        print(f"=========================================")

        # Initialize the task
        task = IDPCNDU()
        task.read_data(d_path)

        # Initialize the offline worker
        worker = offlineQD(task, o_dir, name)

        for seed in range(Configs.REPEAT):
            print(f"\n--- Running Seed {seed} for {name} ---")
            
            # Set seeds for reproducibility
            Configs.rd.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            t1 = time.time()
            
            # Run the offline QD process (handles generations & VAE inference internally)
            best_ind = worker.run(seed)

            duration = time.time() - t1

            # Record stats
            current_cost = -best_ind.fitness
            bf[i] = min(bf[i], current_cost)
            rs[i].append(current_cost)
            time_records[i].append(duration / 60.0) # Store in minutes

            print(f"Task {name} | Seed {seed} | Best distance: {current_cost} | Time: {duration:.2f}s")

        # Calculate and save average stats for this task across all seeds
        avg_val = statistics.mean(rs[i])
        std_val = statistics.stdev(rs[i]) if len(rs[i]) > 1 else 0.0
        avg_time = statistics.mean(time_records[i])

        line = f"{name}\t{int(bf[i])}\t{avg_val:.2f}\t{std_val:.2f}\t{avg_time:.2f} mins\n"
        fw.write(line)
        fw.flush()

def main():
    print("Running VAE-Integrated Solver (Offline Inference Mode)...")
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Inference will run on CPU.")

    # Paths
    output_path = "outputOffline"
    result_file = os.path.join(output_path, "offline_results.txt")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 1. Target a LIST of datasets (add or remove paths here as needed)
    target_files = [
        # "data/idpc_ndu_52_6_204.txt",
        # "data/idpc_ndu_1002_12_82252.txt",
        # "data/idpc_ndu_1002_22_36564.txt",
        # "data/idpc_ndu_2002_17_128021.txt",
        # "data/idpc_ndu_2502_22_229601.txt",
        # "data/idpc_ndu_452_32_10406.txt",
        # "data/idpc_ndu_704_15_16990.txt",
        # "data/idpc_ndu_2802_30_215589.txt",
        # "data/idpc_ndu_3202_42_243376.txt",
        # "data/idpc_ndu_3602_32_406192.txt",
        "data/idpc_ndu_1514_16_78292.txt",
        "data/idpc_ndu_1514_30_78351.txt",
        "data/idpc_ndu_1602_22_60574.txt"
        # "data/idpc_ndu_252_11_3513.txt",
        # "data/idpc_ndu_1256_21_44446.txt",
        # "data/idpc_ndu_102_10_834.txt",
        # "data/idpc_ndu_1730_20_88509.txt",
        # "data/idpc_ndu_2918_29_293109.txt"
    ]

    # 2. Prepare task routing based on the list
    tasks_info = []
    for file_path in target_files:
        if not os.path.exists(file_path):
            print(f"Warning: Dataset not found at '{file_path}'. Skipping.")
            continue
            
        # Extract the name to use for folders and logging
        name = os.path.basename(file_path).replace(".txt", "")
        out_dir = os.path.join(output_path, name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        tasks_info.append((file_path, out_dir, name))

    if not tasks_info:
        print("No valid datasets found from the provided list. Exiting.")
        return

    # 3. Execute tasks and log results
    with open(result_file, "a") as fw:
        # Write header for the summary file if it's newly created
        if os.stat(result_file).st_size == 0:
            fw.write("Dataset\tBest\tAverage\tStdDev\tAvgTime\n")
            
        run_offline_tasks(fw, tasks_info)

    print("\nALL TASKS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()