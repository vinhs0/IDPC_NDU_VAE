import numpy as np
import math
import random
import torch

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
        """
        Algorithm 4: Autoencoder-based Method (AET).
        Adapted for Graph Data (x, edge_index).
        """
        if not target_archive: return None 
        
        # p_data is a PyG Data object
        p_data = random.choice(list(target_archive.values()))
        
        # 1. MATCH DEVICES: Ensure data is on the same hardware as the VAE
        device = next(source_vae.parameters()).device
        x = p_data.x.to(device)
        edge_index = p_data.edge_index.to(device)

        with torch.no_grad():
            if D_t == D_s:
                # Simple reconstruction
                mu, logvar = source_vae.encode(x, edge_index)
                z = source_vae.reparameterize(mu, logvar)
                x_new = source_vae.decode(z, batch_size=1).cpu().numpy()
                
            elif D_t > D_s:
                # Truncate target solution to match source dim (D_s)
                x_trunc = x[:D_s, :]
                
                # Safely remove any edges that refer to nodes we just truncated
                mask = (edge_index[0] < D_s) & (edge_index[1] < D_s)
                edge_index_trunc = edge_index[:, mask]
                
                # Reconstruct the truncated part
                mu, logvar = source_vae.encode(x_trunc, edge_index_trunc)
                z = source_vae.reparameterize(mu, logvar)
                x_rec = source_vae.decode(z, batch_size=1)
                
                # Concatenate reconstructed part with preserved dimensions
                p2 = x[D_s:, :]
                x_new = torch.cat((x_rec, p2), dim=0).cpu().numpy()
                
            else: # D_t < D_s
                # Generate random latent vector z
                z_rand = torch.randn(D_s, source_vae.latent_dim).to(device)
                z_prime = source_vae.decode(z_rand, batch_size=1)
                
                # Get the "excess" dimensions from z_prime
                z_prime_excess = z_prime[D_t:, :] 
                
                # Concatenate target solution features x with excess dimensions
                x_prime = torch.cat((x, z_prime_excess), dim=0)
                
                # Reconstruct this combined vector using Source VAE
                mu, logvar = source_vae.encode(x_prime, edge_index)
                z = source_vae.reparameterize(mu, logvar)
                q = source_vae.decode(z, batch_size=1)
                
                # Truncate back to target dimension D_t
                x_new = q[:D_t, :].cpu().numpy()
            
        return x_new

    def generator_transfer(self, target_archive, source_vae, D_t, D_s):
        """
        Algorithm 5: Generator-based Method (GNT).
        Adapted for Graph Data.
        """
        device = next(source_vae.parameters()).device
        
        # NO_GRAD: Stop PyTorch memory tracking here as well
        with torch.no_grad():
            # Sample from latent distribution N(0,1) for D_s nodes
            z = torch.randn(D_s, source_vae.latent_dim).to(device)
            
            if D_t == D_s:
                # Simple generation
                x_new = source_vae.decode(z, batch_size=1).cpu().numpy()
                
            elif D_t > D_s:
                if not target_archive: return None
                
                p_data = random.choice(list(target_archive.values()))
                x = p_data.x.to(device)
                
                # Decode z to get base structure
                z_prime = source_vae.decode(z, batch_size=1)
                
                # Get excess dimensions from target solution
                p1_excess = x[D_s:, :]
                
                # Concatenate
                x_new = torch.cat((z_prime, p1_excess), dim=0).cpu().numpy()
                
            else: # D_t < D_s
                # Decode z and truncate
                z_prime = source_vae.decode(z, batch_size=1)
                x_new = z_prime[:D_t, :].cpu().numpy()
            
        return x_new

    def perform_transfer(self, target_archive, source_vae, D_t, D_s, batch_size=250):
        """
        Algorithm 3: ARAKT main loop.
        """
        print("Bắt đầu chuyển giao...")
        new_solutions = []
        method = self.select_transfer_method()
        
        for _ in range(batch_size):
            if method == 1:
                sol = self.autoencoder_transfer(target_archive, source_vae, D_t, D_s)
            else:
                sol = self.generator_transfer(target_archive, source_vae, D_t, D_s)
            
            if sol is not None:
                new_solutions.append((method, sol))
                
        print("Kết thúc chuyển giao! Chuyển giao thành công")
        return new_solutions

    def update_bandit(self, results):
        """
        Feedback loop for UCB1.
        """
        for method, reward in results:
            self.update_stats(method, reward)

    def receive_knowledge(self, target_worker, source_vae, source_dim, target_archive_mock, batch_size):
        """
        Orchestrates the reception of knowledge from a foreign VAE into the target worker.
        """
        if source_vae is None:
            return
            
        new_solution_data = self.perform_transfer(
            target_archive=target_archive_mock,
            source_vae=source_vae,
            D_t=target_worker.task_dim,
            D_s=source_dim,
            batch_size=batch_size
        )

        feedback_results = []
        successful_transfers = 0
        for method_idx, new_vec in new_solution_data:
            # Use the target worker's domain knowledge to decode and evaluate
            # kiểm tra thành phần trong vòng lặp for

            new_ind = target_worker.vector_to_individual(new_vec)
            new_ind.update_fitness(target_worker.task)

            # Attempt to add to archive
            was_added, is_new_cell = target_worker.add_to_archive(new_ind)
            
            # Calculate HMQD rewards
            reward = 0
            if is_new_cell:
                reward = 2
                successful_transfers += 1
            elif was_added:
                reward = 1
                successful_transfers += 1
            
            feedback_results.append((method_idx, reward))

        self.update_bandit(feedback_results)
        print(f"Task received transfer. Injected/Improved {successful_transfers}. Bandit Stats: {self.selected}")