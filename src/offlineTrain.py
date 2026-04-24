import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from vae.qdVAE import QD 
from vae.VAEConfigs import Configs
from problem.IDPCNDU import IDPCNDU

def plot_vae_losses(loss_history, task_name, output_dir=None):
    """
    Plots the training loss curves for the Graph VAE and saves it to the output directory.
    """
    epochs = range(1, len(loss_history['total']) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plotting all loss components
    plt.plot(epochs, loss_history['total'], label='Total Loss', color='black', linewidth=2.5)
    plt.plot(epochs, loss_history['node'], label='Node ID Loss', color='red', linestyle='--')
    plt.plot(epochs, loss_history['depth'], label='Depth MSE Loss', color='blue', linestyle='-.')
    plt.plot(epochs, loss_history['kld'], label='KLD Loss', color='green', linestyle=':')
    
    # Formatting the graph
    plt.title(f'VAE Training Loss History: {task_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        save_path = os.path.join(output_dir, f"{task_name}_vae_loss_curve.png")
        plt.savefig(save_path, dpi=300)
        print(f" -> Loss graph saved to: {save_path}")
        
    # Clear the figure from memory so they don't overlap in the loop
    plt.close()

def main():
    print("Running VAE-Integrated Solver (Offline Pre-Training Mode)...")

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training a HUGE VAE on CPU will be extremely slow!")
        return

    output_path = "outputVAE2"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 1. Target a LIST of datasets (add or remove paths here)
    target_files = [
        # "data/idpc_ndu_52_6_204.txt"
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
    
    tasks_info = []
    
    for file_path in target_files:
        if not os.path.exists(file_path):
            print(f"Warning: Dataset not found at '{file_path}'. Skipping.")
            continue
            
        # Extract name without extension for the output folder
        name = os.path.basename(file_path).replace(".txt", "")
        out_dir = os.path.join(output_path, name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        tasks_info.append((file_path, out_dir, name))

    if not tasks_info:
        print("No valid datasets found from the provided list. Exiting.")
        return

    print(f"Found {len(tasks_info)} dataset(s). Initializing Multi-Task QD Flow...")

    # Set fixed seed for reproducibility
    seed = 42
    Configs.rd.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Initialize Workers for the tasks
    workers = []
    for d_path, o_dir, task_name in tasks_info:
        task = IDPCNDU()
        task.read_data(d_path)
        worker = QD(task, o_dir, task_name)
        workers.append(worker)

    # 3. Run MAP-Elites to populate the archive
    generations_to_run = 50
    t1 = time.time()

    print(f"\n--- Phase 1: Running MAP-Elites for {generations_to_run} Generations ---")
    for i, worker in enumerate(workers):
        log_path = os.path.join(tasks_info[i][1], f"{tasks_info[i][2]}_seed({seed}).gen")
        with open(log_path, "w") as fw_gen:
            fw_gen.write(f"Generations {tasks_info[i][2]}\n")

            # Run the evolution batch
            worker.run_batch(generations=generations_to_run, fw_gen=fw_gen, global_gen_start=0)

        # Save the best fitness out to the standard .opt file immediately
        if worker.archive:
            best_ind = max(worker.archive.values(), key=lambda ind: ind.fitness)
            worker.save(seed, best_ind.fitness, t1, time.time())

# 4. Train the MASSIVE VAE models on the generated archives
    print("\n--- Phase 2: Training HUGE GraphVAE Models ---")
    vaes = []
    for worker in workers:
        print(f"\nExtracting Elite Archive and Training VAE for: {worker.file_name}")

        # Call the updated method expecting 4 return values
        try:
            vae_model, elite_genotypes, train_time, loss_history = worker.train_and_get_vae()
        except ValueError:
            print("Error: worker.train_and_get_vae() did not return 4 values. Make sure it returns loss_history!")
            continue

        if vae_model is not None and loss_history is not None:
            print(f" -> Successfully finished massive VAE training in {train_time:.2f} seconds.")
            
            # ==========================================
            # NEW: Save the trained VAE model weights
            # ==========================================
            model_save_path = os.path.join("model_and_data", f"{worker.file_name}_vae.pth")
            torch.save(vae_model.state_dict(), model_save_path)
            print(f" -> Saved VAE model weights to: {model_save_path}")
            # ==========================================
            
            vaes.append(vae_model)
            
            # Plot the losses and save the graph
            plot_vae_losses(loss_history, worker.file_name, output_dir="model_and_data")
            
        else:
            print(f" -> Failed to train VAE (The MAP-Elites archive was likely empty).")
            vaes.append(None)

    total_time = time.time() - t1
    print(f"\nDONE! Processed {len(tasks_info)} dataset(s) and saved massive VAEs in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()