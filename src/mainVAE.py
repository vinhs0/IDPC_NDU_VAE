import os
import time
import statistics
import torch
import numpy as np

from ga.GA import GA
from ga.Configs import Configs
from problem.IDPCNDU import IDPCNDU

from vae.VAE import VAE, train_vae
from vae.KT import KnowledgeTransfer

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
    
    ga = GA(task, output_path, name)
    kt = KnowledgeTransfer()
    bf = float('inf')
    rs = []
    
    for seed in range(Configs.REPEAT):
        Configs.rd.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"Running Seed {seed} for {name}...")
        
        # 1. Run the Optimization (GA)
        # best is an Individual object
        best = ga.run(seed)
        
        # --- HMQD VAE & KT Integration ---
        
        # A. Access the Population
        # We need to find the Population object inside the GA instance.
        # Based on your provided classes, GA likely holds a Population instance.
        # We try common attribute names 'pop' or 'population'.
        pop_obj = getattr(ga, 'pop', None)
        if not pop_obj:
            pop_obj = getattr(ga, 'population', None)
            
        # Ensure we found a Population object and it has individuals
        if pop_obj and hasattr(pop_obj, 'get_population'):
            individuals = pop_obj.get_population()
            
            if individuals and len(individuals) > 0:
                # B. Extract Genotypes
                # The chromosome is a list of NodeDepth objects. 
                # For the VAE, we extract the 'node' ID sequence to form a numerical vector.
                raw_genotypes = []
                for ind in individuals:
                    # ind.get_chromosome() returns List[NodeDepth]
                    chrom = ind.get_chromosome()
                    # Flatten to vector of node IDs: [node1, node2, ...]
                    vec = [nd.node for nd in chrom] 
                    raw_genotypes.append(vec)
                
                # Preprocess: Normalize to [-1, 1]
                elite_genotypes = normalize_data(raw_genotypes)
                
                # Get dimensions
                # shape is (num_individuals, chromosome_length)
                task_dim = elite_genotypes.shape[1]
                
                # [cite_start]C. Train VAE [cite: 316-343]
                # Initialize VAE model (latent_dim=6 as per paper)
                vae_model = VAE(task_dim=task_dim, latent_dim=6)
                
                # Train the model on the current population
                # print(f"Training VAE for seed {seed}...")
                train_vae(vae_model, elite_genotypes, epochs=5)
                
                # [cite_start]D. Perform Knowledge Transfer (Demonstration) [cite: 344-379]
                # In a real Multi-task scenario, 'source_vae' would come from a DIFFERENT task.
                # Here, we demonstrate the call using the current VAE as a self-transfer loop.
                
                # Mocking a target archive (Dict: index -> Genotype Vector)
                target_archive_mock = {i: gen for i, gen in enumerate(elite_genotypes)}
                
                # Run ARAKT (Adaptive Transfer)
                # print("Running Knowledge Transfer...")
                new_solutions = kt.perform_transfer(
                    target_archive=target_archive_mock,
                    source_vae=vae_model,
                    D_t=task_dim, # Target Dimension
                    D_s=task_dim  # Source Dimension
                )
                
                # Feedback loop for Bandit (Updating UCB1 stats)
                # This demonstrates how you would update the bandit based on success
                feedback_results = []
                for method_idx, _ in new_solutions:
                    # Dummy reward logic for demo purposes
                    reward = 1 if np.random.rand() > 0.5 else 0 
                    feedback_results.append((method_idx, reward))
                
                kt.update_bandit(feedback_results)
        
        # ---------------------------------

        # Process Results
        # Fitness is negative cost (e.g. -150), so we negate it to get positive cost (150)
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