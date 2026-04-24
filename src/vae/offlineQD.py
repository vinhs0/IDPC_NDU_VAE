import os
import time
import math
import random
import numpy as np
import torch
from typing import List, Tuple
from torch_geometric.data import Data

from vae.VAEConfigs import Configs
from vae.Population import Population
from vae.Individual import Individual
from vae.NodeDepth import NodeDepth
from vae.Embedder import GraphEmbedder

from vae.VAE2 import GraphVAE, train_vae
from vae.KT2 import KnowledgeTransfer

class offlineQD:
    def __init__(self, task, output_path: str, file_name: str):
        self.task = task
        self.output_path = output_path
        self.out_dir = output_path 
        self.file_name = file_name
        
        # HMQD Components
        self.kt = KnowledgeTransfer()
        self.vae = None 
        
        # MAP-Elites Archive
        self.archive = {}
        self.task_dim = self.task.get_number_of_domains()

    def get_behavior_descriptor(self, ind: Individual) -> Tuple:
        return (ind.domain, )

    def add_to_archive(self, ind: Individual) -> Tuple[bool, bool]:
        if ind.fitness <= -Configs.MAX_VALUE:
            return False, False 

        bd = self.get_behavior_descriptor(ind)

        if bd not in self.archive:
            self.archive[bd] = ind
            return True, True
        
        if ind.fitness > self.archive[bd].fitness:
            self.archive[bd] = ind
            return True, False
            
        return False, False

    def save(self, seed: int, best_fitness: float, t1: float, t2: float):
        out_file_name_opt = f"{self.file_name}_seed({seed}).opt"
        full_path = os.path.join(self.output_path, out_file_name_opt)
        try:
            with open(full_path, "w") as fw_opt:
                fw_opt.write(f"Filename: {self.file_name}\n")
                fw_opt.write(f"Seed: {seed}\n")
                fw_opt.write(f"Fitness: {-best_fitness}\n") 
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

    def vector_to_individual(self, discrete_node_ids, discrete_depths) -> Individual:
        ind = Individual()
        chromosome = []
        
        if torch.is_tensor(discrete_node_ids):
            discrete_node_ids = discrete_node_ids.cpu().numpy()
        if torch.is_tensor(discrete_depths):
            discrete_depths = discrete_depths.cpu().numpy()
            
        for n_id, d in zip(discrete_node_ids, discrete_depths):
            node_id = max(1, min(int(n_id), self.task_dim))
            depth = max(0, int(d))
            chromosome.append(NodeDepth(node_id, depth)) 
            
        ind.set_chromosome(chromosome)
        return ind

    # =========================================================================
    # SINGLE-TASK METHOD: INFERENCE ONLY (No Training)
    # =========================================================================
    def run(self, seed: int) -> Individual:
        """
        Original monolithic run loop, updated to use OFFLINE Pre-Trained VAEs.
        """
        t1 = time.time()
        self.archive.clear()
        
        # 1. Random Initialization
        for _ in range(Configs.POPULATION_SIZE):
            ind = Individual()
            ind.random_init(self.task.adj_domain)
            ind.update_fitness(self.task)
            self.add_to_archive(ind)

        # 2. Pre-Load the VAE Model from your model_and_data folder
        model_path = os.path.join("model_and_data", f"{self.file_name}_vae.pth")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if os.path.exists(model_path):
            print(f"Loading pre-trained VAE from: {model_path}")
            self.vae = GraphVAE(
                num_total_domains=self.task_dim, 
                num_nodes=self.task_dim, 
                latent_dim=8,
                n2v_embedding_dim=15
            )
            # Load weights and set strictly to evaluation mode
            self.vae.load_state_dict(torch.load(model_path, map_location=device))
            self.vae = self.vae.to(device)
            self.vae.eval() 
        else:
            print(f"WARNING: Pre-trained model not found at {model_path}. Knowledge transfer will be disabled!")
            self.vae = None

        generation = 0
        transfer_interval = Configs.TRANSFER_INTERVAL_GEN
        transfer_batch = Configs.TRANSFER_BATCH_SIZE

        out_file_name_gen = f"{self.file_name}_seed({seed}).gen"
        full_path = os.path.join(self.output_path, out_file_name_gen)
        print(f"Logging generations to: {out_file_name_gen}")

        with open(full_path, "w") as fw_gen:
            fw_gen.write(f"Generations {self.file_name}\n")

            while generation < Configs.MAX_GENERATIONS:
                current_best_ind = max(self.archive.values(), key=lambda ind: ind.fitness)
                best_fitness = -current_best_ind.fitness
                
                fw_gen.write(f"{generation} {best_fitness}\n")
                fw_gen.flush()

                archive_parents = list(self.archive.values())
                
                if not archive_parents: 
                    ind = Individual()
                    ind.random_init(self.task.adj_domain)
                    ind.update_fitness(self.task)
                    self.add_to_archive(ind)
                    archive_parents = list(self.archive.values())

                offspring = self.reproduction(archive_parents)
                
                for o in offspring:
                    self.add_to_archive(o)
                
                generation += 1

                # 3. KNOWLEDGE TRANSFER Logic (Inference Only)
                if generation > 0 and generation % transfer_interval == 0 and self.vae is not None:
                    print(f"--- [Gen {generation}] Triggering Inference-Based Transfer ---")
                    current_inds = list(self.archive.values())
                    
                    if len(current_inds) > 0:
                        target_archive_mock = {i: ind for i, ind in enumerate(current_inds)}
                        
                        # FIX: Removed 'with torch.no_grad():' because Node2Vec 
                        # inside perform_transfer requires gradients to train its embeddings!
                        new_solution_data = self.kt.perform_transfer(
                            target_archive=target_archive_mock,
                            source_vae=self.vae,
                            D_t=self.task_dim,
                            D_s=self.task_dim,
                            batch_size=transfer_batch
                        )

                        feedback_results = []
                        successful_transfers = 0

                        for method_idx, new_vec in new_solution_data:
                            new_ind = self.vector_to_individual(new_vec[0], new_vec[1]) 
                            new_ind.update_fitness(self.task)
                                
                            was_added, is_new_cell = self.add_to_archive(new_ind)
                                
                            reward = 0
                            if is_new_cell:
                                reward = 2 
                                successful_transfers += 1
                            elif was_added:
                                reward = 1 
                                successful_transfers += 1
                                
                            feedback_results.append((method_idx, reward))
                            
                        self.kt.update_bandit(feedback_results)
                        print("Done!!")
        
        final_best_ind = max(self.archive.values(), key=lambda ind: ind.fitness)

        t2 = time.time()
        self.save(seed, final_best_ind.fitness, t1, t2) # Removed vae_training_time adjustment
        
        return final_best_ind

    # --- Helpers ---
    def select_parent(self, parents: List[Individual]) -> Individual:
        return random.choice(parents)

    def reproduction(self, parents: List[Individual]) -> List[Individual]:
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