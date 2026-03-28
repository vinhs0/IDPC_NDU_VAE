import os
import time
import math
import numpy as np
import torch
from typing import List

from .VAEConfigs import Configs
from ga.Population import Population
from ga.Individual import Individual
from ga.NodeDepth import NodeDepth 

from VAE import VAE, train_vae
from KT import KnowledgeTransfer

# Luồng GA cũ (bởi vì trong bài này không cần nhiều mục tiêu, nên dùng GA có lẽ là đủ rồi?)

class GA:
    def __init__(self, task, output_path: str, file_name: str):
        self.task = task
        self.output_path = output_path
        self.file_name = file_name
        
        # HMQD Components
        self.kt = KnowledgeTransfer()
        self.vae = None 

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
        
        # Mapping Logic: Scale the continuous range [-1, 1] to the discrete Integer range [1, Num_Domains]
        min_node = 1
        max_node = self.task.get_number_of_domains()
        
        for val in vector:
            # Normalize
            norm_0_1 = (val + 1) / 2
            # Scale to Node ID range
            node_id = int(norm_0_1 * (max_node - min_node) + min_node)
            # Clamp to ensure validity
            node_id = max(min_node, min(node_id, max_node))
            
            # Create Gene (NodeDepth). 
            # Note: Depth is set to 0 initially; the fitness evaluation/decoding 
            # often handles structure, or you might need a local search to fix depths.
            chromosome.append(NodeDepth(node_id, 0)) 
            
        ind.set_chromosome(chromosome)
        return ind

    def save(self, seed: int, pop: Population, t1: float, t2: float):
        out_file_name_opt = f"{self.file_name}_seed({seed}).opt"
        full_path = os.path.join(self.output_path, out_file_name_opt)
        try:
            with open(full_path, "w") as fw_opt:
                fw_opt.write(f"Filename: {self.file_name}\n")
                fw_opt.write(f"Seed: {seed}\n")
                fw_opt.write(f"Fitness: {-pop.get_best_individual().fitness}\n")
                
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
        vae_training_time = 0.0 #Biến tích lũy thời gian train của VAE
        
        population = Population(self.task)
        population.init_population() 
        population.update_best_individual()

        generation = 0

        transfer_interval = Configs.TRANSFER_INTERVAL_GEN
        transfer_batch = Configs.TRANSFER_BATCH_SIZE

        out_file_name_gen = f"{self.file_name}_seed({seed}).gen"
        full_path = os.path.join(self.output_path, out_file_name_gen)
        print(f"Logging generations to: {out_file_name_gen}")

        with open(full_path, "w") as fw_gen:
            fw_gen.write(f"Generations {self.file_name}\n")

            while generation < Configs.MAX_GENERATIONS:
                # Log Current Best
                best_fitness = -population.get_best_individual().fitness
                fw_gen.write(f"{generation} {best_fitness}\n")
                fw_gen.flush()

                # 2. Reproduction (Crossover & Mutation)
                offspring = self.reproduction(population.get_population())
                
                # Prepare intermediate population
                imi_pop = []
                imi_pop.extend(offspring)
                imi_pop.extend(population.get_population())

                # 3. Survival Selection (Standard GA)
                population.set_population(imi_pop)
                population.survival_selection()
                
                # Increment Generation (should be at the end)
                generation += 1

                # KNOWLEDGE TRANSFER (Triggered every 100 Epochs)
                if generation > 0 and generation % transfer_interval == 0:
                    print(f"--- [Gen {generation}] Triggering Knowledge Transfer ---")
                    
                    # A. Data Extraction (Current Elites)
                    current_inds = population.get_population()
                    raw_genotypes = []
                    valid_inds = 0
                    
                    for ind in current_inds:
                        chrom = ind.get_chromosome()
                        if chrom: # Ensure chromosome is not empty
                            vec = [nd.node for nd in chrom]
                            raw_genotypes.append(vec)
                            valid_inds += 1
                    
                    if valid_inds > 0:
                        # Normalize inputs for VAE [-1, 1]
                        elite_genotypes = self.normalize_data(raw_genotypes)
                        task_dim = elite_genotypes.shape[1]

                        # VAE training (ở đây có đếm thời gian training để trừ bớt về sau)
                        self.vae = VAE(task_dim=task_dim, latent_dim=6)
                        start_train_time = time.time()
                        train_vae(self.vae, elite_genotypes, epochs=5)
                        vae_training_time += (time.time() - start_train_time)

                        # C. Perform Transfer
                        # Mock Target Archive = Current Population Map
                        target_archive_mock = {i: gen for i, gen in enumerate(elite_genotypes)}
                        
                        new_solution_data = self.kt.perform_transfer(
                            target_archive=target_archive_mock,
                            source_vae=self.vae,
                            D_t=task_dim,
                            D_s=task_dim,
                            batch_size=transfer_batch
                        )

                        # D. Inject New Solutions
                        transfer_candidates = []
                        feedback_results = []
                        current_best_fit = population.get_best_individual().fitness

                        for method_idx, new_vec in new_solution_data:
                            # Convert Vector -> Individual
                            new_ind = self.vector_to_individual(new_vec)
                            
                            # Evaluate
                            new_ind.update_fitness(self.task)
                            
                            # Add to a temporary list to merge later
                            transfer_candidates.append(new_ind)

                            # Calculate Reward (Simple improvement check)
                            reward = 0
                            if new_ind.fitness > current_best_fit:
                                reward = 1  # Fitness Improvement
                            
                            feedback_results.append((method_idx, reward))
                        
                        # Update Bandit Stats
                        self.kt.update_bandit(feedback_results)

                        # E. Merge & Reselect
                        current_pop = population.get_population()
                        current_pop.extend(transfer_candidates)
                        population.set_population(current_pop)
                        population.survival_selection()
                        
                        print(f"    -> Injected {len(transfer_candidates)} solutions. Bandit Stats: {self.kt.selected}")
                # ============================================================
        # Thời gian chạy (không tính thời gian train VAE)
        t2 = time.time()
        adjusted_t2 = t2 - vae_training_time 
        self.save(seed, population, t1, adjusted_t2)
        
        return population.get_best_individual()

    def select_parent(self, parents: List[Individual]) -> Individual:
        pos1 = Configs.rd.randint(0, len(parents) - 1)
        pos2 = Configs.rd.randint(0, len(parents) - 1)
        while pos1 == pos2:
            pos2 = Configs.rd.randint(0, len(parents) - 1)
        p1 = parents[pos1]
        p2 = parents[pos2]
        return p1 if p1.fitness > p2.fitness else p2

    def reproduction(self, parents: List[Individual]) -> List[Individual]:
        offspring_pop = Population(self.task)
        current_offspring = offspring_pop.get_population() 
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