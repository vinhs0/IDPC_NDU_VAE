import os
import time
import math
import random
import numpy as np
import torch
from typing import List, Tuple

from .VAEConfigs import Configs
from ga.Population import Population
from ga.Individual import Individual
from ga.NodeDepth import NodeDepth 

from .VAE import VAE, train_vae
from .KT import KnowledgeTransfer

class QD:
    def __init__(self, task, output_path: str, file_name: str):
        self.task = task
        self.output_path = output_path
        self.file_name = file_name
        
        # HMQD Components
        self.kt = KnowledgeTransfer()
        self.vae = None 
        
        # MAP-Elites Archive
        self.archive = {}

    def get_behavior_descriptor(self, ind: Individual) -> Tuple:
        """
        Defines the Behavior Descriptor (BD) to index the MAP-Elites grid.
        For routing/graph problems, the number of domains traversed is a common BD.
        """
        return (ind.domain, )

    def add_to_archive(self, ind: Individual) -> Tuple[bool, bool]:
        """
        MAP-Elites insertion logic.
        Returns: (was_added, is_new_cell)
        """
        if ind.fitness <= -Configs.MAX_VALUE:
            return False, False # Invalid solution

        bd = self.get_behavior_descriptor(ind)

        # If cell is empty, add it (Reward +2 scenario in HMQD)
        if bd not in self.archive:
            self.archive[bd] = ind
            return True, True
        
        # If cell is occupied, only replace if the new fitness is strictly better (Reward +1 scenario)
        if ind.fitness > self.archive[bd].fitness:
            self.archive[bd] = ind
            return True, False
            
        return False, False

    def normalize_data(self, data):
        """Helper to normalize data to [-1, 1] for VAE (Tanh output)."""
        data = np.array(data)
        if data.size == 0: return data
        min_val, max_val = np.min(data), np.max(data)
        if max_val - min_val == 0: return data
        return 2 * (data - min_val) / (max_val - min_val) - 1

    def vector_to_individual(self, vector: np.array) -> Individual:
        """
        Converts VAE output (continuous vector [-1, 1]) back to Individual (discrete Node IDs).
        """
        ind = Individual()
        chromosome = []
        
        min_node = 1
        max_node = self.task.get_number_of_domains()
        
        for val in vector:
            norm_0_1 = (val + 1) / 2
            node_id = int(norm_0_1 * (max_node - min_node) + min_node)
            node_id = max(min_node, min(node_id, max_node))
            chromosome.append(NodeDepth(node_id, 0)) 
            
        ind.set_chromosome(chromosome)
        return ind

    def save(self, seed: int, best_fitness: float, t1: float, t2: float):
        out_file_name_opt = f"{self.file_name}_seed({seed}).opt"
        full_path = os.path.join(self.output_path, out_file_name_opt)
        try:
            with open(full_path, "w") as fw_opt:
                fw_opt.write(f"Filename: {self.file_name}\n")
                fw_opt.write(f"Seed: {seed}\n")
                fw_opt.write(f"Fitness: {-best_fitness}\n") # Saving as positive cost
                fw_opt.write(f"Archive Coverage: {len(self.archive)} cells\n")
                
                duration_sec = t2 - t1
                hours = int(duration_sec // 3600)
                minutes = int((duration_sec % 3600) // 60)
                seconds = int(duration_sec % 60)
                milliseconds = int((duration_sec * 1000) % 1000)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
                
                fw_opt.write(f"Time: {time_str}\n")
                fw_opt.flush()
        except IOError as e:
            print(f"Error saving file: {e}")

    def run(self, seed: int) -> Individual:
        t1 = time.time()
        vae_training_time = 0.0 # Biến tích lũy thời gian train của VAE
        
        # Initialize MAP-Elites Archive
        self.archive.clear()
        
        # 1. Random Initialization
        for _ in range(Configs.POPULATION_SIZE):
            ind = Individual()
            ind.random_init(self.task.adj_domain)
            ind.update_fitness(self.task)
            self.add_to_archive(ind)

        generation = 0
        transfer_interval = Configs.TRANSFER_INTERVAL_GEN
        transfer_batch = Configs.TRANSFER_BATCH_SIZE

        out_file_name_gen = f"{self.file_name}_seed({seed}).gen"
        full_path = os.path.join(self.output_path, out_file_name_gen)
        print(f"Logging generations to: {out_file_name_gen}")

        with open(full_path, "w") as fw_gen:
            fw_gen.write(f"Generations {self.file_name}\n")

            while generation < Configs.MAX_GENERATIONS:
                
                # Get current global best from archive for logging
                current_best_ind = max(self.archive.values(), key=lambda ind: ind.fitness)
                best_fitness = -current_best_ind.fitness
                
                fw_gen.write(f"{generation} {best_fitness}\n")
                fw_gen.flush()

                # 2. Reproduction (Crossover & Mutation)
                # We pass the archive values as the parent pool
                archive_parents = list(self.archive.values())
                
                # Fallback if archive is somehow empty
                if not archive_parents: 
                    ind = Individual()
                    ind.random_init(self.task.adj_domain)
                    ind.update_fitness(self.task)
                    self.add_to_archive(ind)
                    archive_parents = list(self.archive.values())

                offspring = self.reproduction(archive_parents)
                
                # 3. Grid Assignment (MAP-Elites Replacement)
                # Instead of standard survival_selection(), we try to insert each offspring into the grid
                for o in offspring:
                    self.add_to_archive(o)
                
                generation += 1

                # 4. KNOWLEDGE TRANSFER (Triggered every X Epochs)
                if generation > 0 and generation % transfer_interval == 0:
                    print(f"--- [Gen {generation}] Triggering Knowledge Transfer ---")
                    
                    # A. Data Extraction (Current Elites directly from the Archive)
                    current_inds = list(self.archive.values())
                    raw_genotypes = []
                    
                    for ind in current_inds:
                        chrom = ind.get_chromosome()
                        if chrom:
                            vec = [nd.node for nd in chrom]
                            raw_genotypes.append(vec)
                    
                    if len(raw_genotypes) > 0:
                        # Normalize inputs for VAE [-1, 1]
                        elite_genotypes = self.normalize_data(raw_genotypes)
                        task_dim = elite_genotypes.shape[1]

                        # B. Train VAE (đếm thời gian training để trừ bớt về sau)
                        self.vae = VAE(task_dim=task_dim, latent_dim=6)
                        start_train_time = time.time()
                        train_vae(self.vae, elite_genotypes, epochs=5)
                        vae_training_time += (time.time() - start_train_time)

                        # C. Perform Transfer
                        target_archive_mock = {i: gen for i, gen in enumerate(elite_genotypes)}
                        
                        new_solution_data = self.kt.perform_transfer(
                            target_archive=target_archive_mock,
                            source_vae=self.vae,
                            D_t=task_dim,
                            D_s=task_dim,
                            batch_size=transfer_batch
                        )

                        # D. Inject New Solutions & Update UCB1 Bandit
                        feedback_results = []
                        successful_transfers = 0

                        for method_idx, new_vec in new_solution_data:
                            # Convert Vector -> Individual
                            new_ind = self.vector_to_individual(new_vec)
                            new_ind.update_fitness(self.task)
                            
                            # Attempt to add to MAP-Elites Archive
                            was_added, is_new_cell = self.add_to_archive(new_ind)
                            
                            # Calculate Reward based on HMQD paper rules
                            reward = 0
                            if is_new_cell:
                                reward = 2  # Discovered a completely new grid cell
                                successful_transfers += 1
                            elif was_added:
                                reward = 1  # Improved an existing grid cell
                                successful_transfers += 1
                            
                            feedback_results.append((method_idx, reward))
                        
                        # Update Bandit Stats
                        self.kt.update_bandit(feedback_results)
                        
                        print(f"    -> Successfully injected/improved {successful_transfers} solutions. Bandit Stats: {self.kt.selected}")
                # ============================================================
        
        # Get absolute best individual before exiting
        final_best_ind = max(self.archive.values(), key=lambda ind: ind.fitness)

        # Thời gian chạy (không tính thời gian train VAE)
        t2 = time.time()
        adjusted_t2 = t2 - vae_training_time 
        self.save(seed, final_best_ind.fitness, t1, adjusted_t2)
        
        return final_best_ind

    # --- Helpers ---
    def select_parent(self, parents: List[Individual]) -> Individual:
        """
        In standard MAP-Elites, parent selection is uniformly random across the archive.
        """
        return random.choice(parents)

    def reproduction(self, parents: List[Individual]) -> List[Individual]:
        # Using the Population class purely to access the crossover/mutation methods
        offspring_pop = Population(self.task)
        current_offspring = [] 
        
        while len(current_offspring) < Configs.POPULATION_SIZE:
            p1 = self.select_parent(parents)
            p2 = self.select_parent(parents)
            
            if Configs.rd.random() < Configs.CROSSOVER_RATE:
                o = offspring_pop.crossover(p1, p2)
                if Configs.rd.random() < Configs.MUTATION_RATE:
                    o = offspring_pop.mutation(o)
                    
                o.update_fitness(self.task)
                current_offspring.append(o)
                
        return current_offspring