import os
import time
import math
import random
import numpy as np
import torch
from typing import List, Tuple
from torch_geometric.data import Data

from .VAEConfigs import Configs
from vae.Population import Population
from vae.Individual import Individual
from vae.NodeDepth import NodeDepth
from vae.Embedder import GraphEmbedder

from .VAE2 import GraphVAE, train_vae
from .KT2 import KnowledgeTransfer

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
        self.task_dim = self.task.get_number_of_domains()

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

    # def normalize_data(self, data):
    #     """Helper to normalize data to [-1, 1] for VAE (Tanh output)."""
    #     data = np.array(data)
    #     if data.size == 0: return data
    #     min_val, max_val = np.min(data), np.max(data)
    #     if max_val - min_val == 0: return data
    #     return 2 * (data - min_val) / (max_val - min_val) - 1

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


    def run_batch(self, generations: int, fw_gen=None, global_gen_start: int = 0):
        """
        Chạy MAP_Elites evolution cho 1 lượng generation nhất định, sau đó dừng để chuyển giao tri thức
        (chuyển giao tri thức - knowledge transfer được thực hiện periodically)
        """
        # Ensure archive is seeded if starting fresh
        if not self.archive:
            for _ in range(Configs.POPULATION_SIZE):
                ind = Individual()
                ind.random_init(self.task.adj_domain)
                ind.update_fitness(self.task)
                self.add_to_archive(ind)
        print(f"Running QD algo for {self.file_name}")
        for g in range(generations):
            print(f"Current generation: {g}")
            current_best_ind = max(self.archive.values(), key=lambda ind: ind.fitness)
            best_fitness = -current_best_ind.fitness
            
            if fw_gen:
                fw_gen.write(f"{global_gen_start + g} {best_fitness}\n")
                fw_gen.flush()

            archive_parents = list(self.archive.values())
            
            if not archive_parents: 
                ind = Individual()
                ind.random_init(self.task.adj_domain)
                ind.update_fitness(self.task)
                self.add_to_archive(ind)
                archive_parents = list(self.archive.values())
            print(f"Fitness: {best_fitness}")
            offspring = self.reproduction(archive_parents)
            for o in offspring:
                self.add_to_archive(o)

    def vector_to_individual(self, discrete_node_ids, discrete_depths) -> Individual:
        """
        đổi discrete node thành List[NodeDepth]
        """
        ind = Individual()
        chromosome = []
        
        # Ensure we are working with flat numpy arrays
        if torch.is_tensor(discrete_node_ids):
            discrete_node_ids = discrete_node_ids.cpu().numpy()
        if torch.is_tensor(discrete_depths):
            discrete_depths = discrete_depths.cpu().numpy()
            
        for n_id, d in zip(discrete_node_ids, discrete_depths):
            # Clamp node ID to valid range (1 to max_node) to prevent classification anomalies
            node_id = max(1, min(int(n_id), self.task_dim))
            
            # Ensure depth doesn't go below 0
            depth = max(0, int(d))
            
            chromosome.append(NodeDepth(node_id, depth)) 
            
        ind.set_chromosome(chromosome)
        return ind

    # ... [Keep your save, run_batch, and helpers exactly the same] ...


    def train_and_get_vae(self): 
        """
        Extracts elites, prepares PyTorch Geometric graph data, 
        trains the GraphVAE, and returns it along with loss history.
        """
        current_inds = list(self.archive.values())
        
        if not current_inds:
            print("No archival solutions found. Exiting...")
            # FIX: Return 4 items to prevent unpacking errors
            return None, [], 0.0, None
            
        graph_data_list = GraphVAE.prepare_graph_data(current_inds)
        
        if len(graph_data_list) == 0:
            # FIX: Return 4 items to prevent unpacking errors
            return None, [], 0.0, None

        vae_model = GraphVAE(
            num_total_domains=self.task_dim, 
            num_nodes=self.task_dim, 
            latent_dim=8,
            n2v_embedding_dim=15
        )
        
        # Ensure model is pushed to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae_model = vae_model.to(device)
        
        start_train_time = time.time()
        
        # FIX: Catch the loss_history returned by your updated train_vae function
        vae_model, loss_history = train_vae(vae_model, graph_data_list, epochs=50)
        
        training_time = time.time() - start_train_time
        
        # FIX: Return all 4 items expected by your main() script
        return vae_model, current_inds, training_time, loss_history
    # ---------------- OLD REPLACED METHOD ----------------
    # def train_and_get_vae(self): 
    #     """
    #     Extracts elites, prepares PyTorch Geometric graph data, 
    #     trains the GraphVAE, and returns it.
    #     """
    #     current_inds = list(self.archive.values())
        
    #     if not current_inds:
    #         print("No archival solutions found. Exiting...")
    #         return None, [], 0.0
            
    #     graph_data_list = GraphVAE.prepare_graph_data(current_inds)
        
    #     if len(graph_data_list) == 0:
    #         return None, [], 0.0

    #     vae_model = GraphVAE(
    #         num_total_domains=self.task_dim, 
    #         num_nodes=self.task_dim, 
    #         latent_dim=8,
    #         n2v_embedding_dim=15
    #     )
        
    #     # FIX: Ensure model is pushed to GPU if available
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     vae_model = vae_model.to(device)
        
    #     start_train_time = time.time()
    #     # Train on the same device
    #     train_vae(vae_model, graph_data_list, epochs=50)
    #     training_time = time.time() - start_train_time
        
    #     return vae_model, graph_data_list, training_time

    # def train_and_get_vae(self): 
    #     """
    #     Extracts elites, prepares PyTorch Geometric graph data, 
    #     trains the GraphVAE, and returns it.
    #     """
    #     current_inds = list(self.archive.values())
        
    #     if not current_inds:
    #         print("No archival solutions found. Exiting...")
    #         # FIX: Must return 3 items to match the unpacking in mainVAE.py
    #         return None, [], 0.0
            
    #     graph_data_list = GraphVAE.prepare_graph_data(current_inds)
        
    #     if len(graph_data_list) == 0:
    #         return None, [], 0.0

    #     vae_model = GraphVAE(node_feature_dim=2, num_nodes=self.task_dim, latent_dim=6)
        
    #     start_train_time = time.time()
    #     train_vae(vae_model, graph_data_list, epochs=10)
    #     training_time = time.time() - start_train_time
        
    #     return vae_model, graph_data_list, training_time

    # =========================================================================
    # SINGLE-TASK METHOD (Restored monolithic run loop for backward compatibility)
    # =========================================================================

    # def run(self, seed: int) -> Individual:
    #     """
    #     Original monolithic run loop. Performs self-transfer for single tasks.
    #     """
    #     t1 = time.time()
    #     vae_training_time = 0.0 
        
    #     self.archive.clear()
        
    #     # Random Initialization
    #     for _ in range(Configs.POPULATION_SIZE):
    #         ind = Individual()
    #         ind.random_init(self.task.adj_domain)
    #         ind.update_fitness(self.task)
    #         self.add_to_archive(ind)

    #     generation = 0
    #     transfer_interval = Configs.TRANSFER_INTERVAL_GEN
    #     transfer_batch = Configs.TRANSFER_BATCH_SIZE

    #     out_file_name_gen = f"{self.file_name}_seed({seed}).gen"
    #     full_path = os.path.join(self.output_path, out_file_name_gen)
    #     print(f"Logging generations to: {out_file_name_gen}")

    #     with open(full_path, "w") as fw_gen:
    #         fw_gen.write(f"Generations {self.file_name}\n")

    #         while generation < Configs.MAX_GENERATIONS:
    #             current_best_ind = max(self.archive.values(), key=lambda ind: ind.fitness)
    #             best_fitness = -current_best_ind.fitness
                
    #             fw_gen.write(f"{generation} {best_fitness}\n")
    #             fw_gen.flush()

    #             archive_parents = list(self.archive.values())
                
    #             if not archive_parents: 
    #                 ind = Individual()
    #                 ind.random_init(self.task.adj_domain)
    #                 ind.update_fitness(self.task)
    #                 self.add_to_archive(ind)
    #                 archive_parents = list(self.archive.values())

    #             offspring = self.reproduction(archive_parents)
                
    #             for o in offspring:
    #                 self.add_to_archive(o)
                
    #             generation += 1

    #             # SELF-TRANSFER Logic
    #             if generation > 0 and generation % transfer_interval == 0:
    #                 print(f"--- [Gen {generation}] Triggering Self-Transfer Knowledge ---")
    #                 current_inds = list(self.archive.values())
                    
    #                 # Uses the updated PyG static extractor
    #                 graph_data_list = GraphVAE.prepare_graph_data(current_inds)
                    
    #                 if len(graph_data_list) > 0:
    #                     self.vae = GraphVAE(node_feature_dim=2, num_nodes=self.task_dim, latent_dim=6)
    #                     start_train_time = time.time()
    #                     train_vae(self.vae, graph_data_list, epochs=5)
    #                     vae_training_time += (time.time() - start_train_time)

    #                     # Provide graph data list as the target mock
    #                     target_archive_mock = {i: data for i, data in enumerate(graph_data_list)}
                        
    #                     new_solution_data = self.kt.perform_transfer(
    #                         target_archive=target_archive_mock,
    #                         source_vae=self.vae,
    #                         D_t=self.task_dim,
    #                         D_s=self.task_dim,
    #                         batch_size=transfer_batch
    #                     )

    #                     feedback_results = []
    #                     successful_transfers = 0

    #                     for method_idx, new_vec in new_solution_data:
    #                         new_ind = self.vector_to_individual(new_vec)
    #                         new_ind.update_fitness(self.task)
                            
    #                         was_added, is_new_cell = self.add_to_archive(new_ind)
                            
    #                         reward = 0
    #                         if is_new_cell:
    #                             reward = 2 
    #                             successful_transfers += 1
    #                         elif was_added:
    #                             reward = 1 
    #                             successful_transfers += 1
                            
    #                         feedback_results.append((method_idx, reward))
                        
    #                     self.kt.update_bandit(feedback_results)
    #                     print(f"    -> Successfully injected/improved {successful_transfers} solutions. Bandit Stats: {self.kt.selected}")
        
        # final_best_ind = max(self.archive.values(), key=lambda ind: ind.fitness)

        # t2 = time.time()
        # adjusted_t2 = t2 - vae_training_time 
        # self.save(seed, final_best_ind.fitness, t1, adjusted_t2)
        
        # return final_best_ind

    # --- Helpers ---
    def select_parent(self, parents: List[Individual]) -> Individual:
        """In standard MAP-Elites, parent selection is uniformly random across the archive."""
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