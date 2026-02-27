import os
import time
import statistics
import torch
import numpy as np

from vae.qdVAE import QD
from ga.Configs import Configs
from problem.IDPCNDU import IDPCNDU

def normalize_data(data):
    """
    Helper to normalize data to [-1, 1] for VAE (Tanh activation).
    """
    data = np.array(data)
    if data.size == 0:
        return data
        
    min_val = np.min(data)
    max_val = np.max(data)
    
    if max_val - min_val == 0:
        return data
        
    return 2 * (data - min_val) / (max_val - min_val) - 1

def solver(fw, data_path, output_path, name):
    t1 = time.time()
    task = IDPCNDU()
    task.read_data(data_path)
    
    qd = QD(task, output_path, name)
    bf = float('inf')
    rs = []
    
    for seed in range(Configs.REPEAT):
        Configs.rd.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"Running Seed {seed} for {name}...")
        
        # Chay thuat toan QD
        best = qd.run(seed)

        current_cost = -best.fitness
        print(f"Seed {seed} Best distance: {current_cost}")
        
        bf = min(current_cost, bf)
        rs.append(current_cost)
        print("---------------------------------------")
    
    t2 = time.time()
    
    # Calculate statistics
    total_duration_minutes = (t2 - t1) / 60
    avg_time_per_run = total_duration_minutes / Configs.REPEAT
    
    avg_val = statistics.mean(rs)
    std_val = statistics.stdev(rs) if len(rs) > 1 else 0.0
    
    # Write to result file
    # Format: Name \t BF \t AVG \t STD \t Time
    line = f"{name}\t{int(bf)}\t{avg_val:.2f}\t{std_val:.2f}\t{avg_time_per_run:.2f}\n"
    fw.write(line)
    fw.flush()

def build_model(fw, file_path, data_path):
    input_list_file = os.path.join(file_path, "file.txt")
    
    if not os.path.exists(input_list_file):
        print(f"Warning: {input_list_file} not found.")
        return

    with open(input_list_file, 'r') as sc:
        # Skip header line
        header = sc.readline()
        
        for line in sc:
            name = line.strip()
            if not name:
                continue
            
            # Create output subdirectory for this instance
            out_dir = os.path.join(file_path, name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
            # Construct data path
            dt_path = os.path.join(data_path, name + ".txt")
            
            solver(fw, dt_path, out_dir, name)

def main():
    print("Running VAE-Integrated Solver...")
    print("Is Torch available?", torch.cuda.is_available())
    if (not torch.cuda.is_available()):
        print("Config CUDA first!, exiting...")
        return 
    
    data_path = "D:\\multi-nde-py\\data" 
    output_path = "outputVAE"
    result_file = os.path.join("outputVAE", "test_vae_results.txt")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    children = []
    if os.path.exists(output_path):
        for entry in os.scandir(output_path):
            if entry.is_dir():
                children.append(entry)

    def sort_key(entry):
        try:
            return entry.name.split("_")[1]
        except IndexError:
            return entry.name

    children.sort(key=sort_key, reverse=True)
    
    with open(result_file, "a") as fw:
        for entry in children:
            file_path = entry.path
            
            # Basic check to skip non-directory files
            if not entry.is_dir():
                continue
            
            print(f"Processing directory: {entry.name}")
            build_model(fw, file_path, data_path)
            
    print("DONE!")

if __name__ == "__main__":
    main()