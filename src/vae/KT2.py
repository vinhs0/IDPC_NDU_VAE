import numpy as np
import math
import random
import torch

from vae.VAE2 import GraphVAE

# get_discrete_node_features() la gi???

class KnowledgeTransfer:
    def __init__(self):
        # UCB1 Statistics for bandit selection
        # 1: Autoencoder-based (AET), 2: Generator-based (GNT)
        self.selected = {1: 0, 2: 0}
        self.successes = {1: 0.0, 2: 0.0}
        self.total_selections = 0

    def select_transfer_method(self):
        """
        Selects transfer method using UCB1 (Eq. 6).
        Returns 1 for AET, 2 for GNT.
        """
        if self.selected[1] == 0: return 1
        if self.selected[2] == 0: return 2

        def ucb_score(method):
            avg_reward = self.successes[method] / self.selected[method]
            exploration = math.sqrt(2 * math.log(self.total_selections) / self.selected[method])
            return avg_reward + exploration

        score_aet = ucb_score(1)
        score_gnt = ucb_score(2)

        return 1 if score_aet >= score_gnt else 2

    def update_stats(self, method, reward):
        """
        Updates UCB1 stats.
        """
        self.selected[method] += 1
        self.successes[method] += reward
        self.total_selections += 1

    def autoencoder_transfer(self, target_archive, source_vae, D_t, D_s):
        if not target_archive: return None 
        
        target_ind = random.choice(list(target_archive.values()))
        
        # Prepare Graph Data for the target individual
        # Note: target_archive holds Individual objects, we use GraphVAE's prep method
        # to generate the exact input features (x, edge_index) needed 
        p_data = GraphVAE.prepare_graph_data([target_ind])[0]
        
        device = next(source_vae.parameters()).device
        # Autoencoder khac cai generator o day
        x = p_data.x.to(device)
        edge_index = p_data.edge_index.to(device)
        max_depth = p_data.max_depth_val.item()

        with torch.no_grad():
            if D_t == D_s:
                mu, logvar = source_vae.encode(x, edge_index)
                z = source_vae.reparameterize(mu, logvar)
                recon_depths, recon_logits = source_vae.decode(z, batch_size=1)
                
                final_nodes, final_depths = source_vae.get_discrete_node_features(recon_depths, recon_logits, max_depth)
                
            elif D_t > D_s:
                x_trunc = x[:D_s, :]
                mask = (edge_index[0] < D_s) & (edge_index[1] < D_s)
                edge_index_trunc = edge_index[:, mask]
                
                mu, logvar = source_vae.encode(x_trunc, edge_index_trunc)
                z = source_vae.reparameterize(mu, logvar)
                recon_depths, recon_logits = source_vae.decode(z, batch_size=1)
                
                rec_nodes, rec_depths = source_vae.get_discrete_node_features(recon_depths, recon_logits, max_depth)
                
                # Append the "excess" nodes directly from the target Individual's original chromosome
                excess_nodes = torch.tensor([nd.node for nd in target_ind.chromosome[D_s:]], device=device)
                excess_depths = torch.tensor([nd.depth for nd in target_ind.chromosome[D_s:]], device=device)
                
                final_nodes = torch.cat((rec_nodes, excess_nodes), dim=0)
                final_depths = torch.cat((rec_depths, excess_depths), dim=0)
                
            else: # D_t < D_s
                z_rand = torch.randn(D_s, source_vae.latent_dim).to(device)
                z_prime_depths, z_prime_logits = source_vae.decode(z_rand, batch_size=1)
                
                # Truncate the discrete outputs back to D_t
                rec_nodes, rec_depths = source_vae.get_discrete_node_features(z_prime_depths, z_prime_logits, max_depth)
                final_nodes = rec_nodes[:D_t]
                final_depths = rec_depths[:D_t]
                
        return final_nodes.cpu().numpy(), final_depths.cpu().numpy() # Gửi về cpu

    def generator_transfer(self, target_archive, source_vae, D_t, D_s):
        device = next(source_vae.parameters()).device
        
        with torch.no_grad():
            z = torch.randn(D_s, source_vae.latent_dim).to(device)
            recon_depths, recon_logits = source_vae.decode(z, batch_size=1)
            
            # Assume a default max_depth of 10 if we don't have a target archive context
            max_depth = 10.0 
            
            if target_archive:
                target_ind = random.choice(list(target_archive.values()))
                max_depth = max([nd.depth for nd in target_ind.chromosome])
            
            rec_nodes, rec_depths = source_vae.get_discrete_node_features(recon_depths, recon_logits, max_depth)
            
            if D_t == D_s:
                final_nodes, final_depths = rec_nodes, rec_depths
                
            elif D_t > D_s:
                if not target_archive: return None
                
                excess_nodes = torch.tensor([nd.node for nd in target_ind.chromosome[D_s:]], device=device)
                excess_depths = torch.tensor([nd.depth for nd in target_ind.chromosome[D_s:]], device=device)
                
                final_nodes = torch.cat((rec_nodes, excess_nodes), dim=0)
                final_depths = torch.cat((rec_depths, excess_depths), dim=0)
                
            else: # D_t < D_s
                final_nodes = rec_nodes[:D_t]
                final_depths = rec_depths[:D_t]
            
        return final_nodes.cpu().numpy(), final_depths.cpu().numpy()
        
    def receive_knowledge(self, target_worker, source_vae, source_dim, target_archive_mock, batch_size):
        if source_vae is None: return
            
        new_solution_data = self.perform_transfer(
            target_archive=target_worker.archive, # Pass the actual archive dictionary
            source_vae=source_vae,
            D_t=target_worker.task_dim,
            D_s=source_dim,
            batch_size=batch_size
        )

        feedback_results = []
        successful_transfers = 0
        
        for method_idx, (discrete_nodes, discrete_depths) in new_solution_data:
            # Pass BOTH discrete arrays to the worker
            new_ind = target_worker.vector_to_individual(discrete_nodes, discrete_depths)
            new_ind.update_fitness(target_worker.task)

            was_added, is_new_cell = target_worker.add_to_archive(new_ind)
            
            reward = 0
            if is_new_cell:
                reward = 2
                successful_transfers += 1
            elif was_added:
                reward = 1
                successful_transfers += 1
            
            feedback_results.append((method_idx, reward))

        self.update_bandit(feedback_results)
        task_identifier = getattr(target_worker, 'file_name', 'Unknown')
        # print(f"    Task [{task_identifier}] received transfer. Injected/Improved {successful_transfers}. Bandit Stats: {self.selected}")

    def perform_transfer(self, target_archive, source_vae, D_t, D_s, batch_size=250):
        """
        Algorithm 3: ARAKT main loop.
        """
        # print("Bắt đầu chuyển giao...")
        new_solutions = []
        method = self.select_transfer_method()
        
        for _ in range(batch_size):
            if method == 1:
                sol = self.autoencoder_transfer(target_archive, source_vae, D_t, D_s)
            else:
                sol = self.generator_transfer(target_archive, source_vae, D_t, D_s)
            
            if sol is not None:
                new_solutions.append((method, sol))
                
        # print("Kết thúc chuyển giao! Chuyển giao thành công")
        return new_solutions

    def update_bandit(self, results):
        """
        Feedback loop for UCB1.
        """
        for method, reward in results:
            self.update_stats(method, reward)

    # def receive_knowledge(self, target_worker, source_vae, source_dim, target_archive_mock, batch_size):
    #     """
    #     Orchestrates the reception of knowledge from a foreign VAE into the target worker.
    #     """
    #     if source_vae is None:
    #         return
            
    #     new_solution_data = self.perform_transfer(
    #         target_archive=target_archive_mock,
    #         source_vae=source_vae,
    #         D_t=target_worker.task_dim,
    #         D_s=source_dim,
    #         batch_size=batch_size
    #     )

    #     feedback_results = []
    #     successful_transfers = 0
    #     for method_idx, new_vec in new_solution_data:
    #         # Use the target worker's domain knowledge to decode and evaluate
    #         new_ind = target_worker.vector_to_individual(new_vec)
    #         new_ind.update_fitness(target_worker.task)

    #         # Attempt to add to archive
    #         was_added, is_new_cell = target_worker.add_to_archive(new_ind)
            
    #         # Calculate HMQD rewards
    #         reward = 0
    #         if is_new_cell:
    #             reward = 2
    #             successful_transfers += 1
    #         elif was_added:
    #             reward = 1
    #             successful_transfers += 1
            
    #         feedback_results.append((method_idx, reward))

    #     self.update_bandit(feedback_results)
    #     # print(f"Task received transfer. Injected/Improved {successful_transfers}. Bandit Stats: {self.selected}")