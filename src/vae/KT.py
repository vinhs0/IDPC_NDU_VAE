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
        Selects transfer method using UCB1 (Eq. 6)[cite: 497].
        Returns 1 for AET, 2 for GNT.
        """
        # Initial exploration: try both at least once
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
        Reward: +1 for fitness improvement, +2 for new grid[cite: 378].
        """
        self.selected[method] += 1
        self.successes[method] += reward
        self.total_selections += 1

    def autoencoder_transfer(self, target_archive, source_vae, D_t, D_s):
        """
        Algorithm 4: Autoencoder-based Method (AET)[cite: 382].
        """
        # Randomly select a solution p from target archive
        if not target_archive: return None # Handle empty archive case
        p = random.choice(list(target_archive.values()))
        p_tensor = torch.FloatTensor(p).unsqueeze(0)

        # Logic for dimension mismatch [cite: 440-454]
        if D_t == D_s:
            # Simple reconstruction
            mu, logvar = source_vae.encode(p_tensor)
            z = source_vae.reparameterize(mu, logvar)
            x_new = source_vae.decode(z).detach().numpy().flatten()
            
        elif D_t > D_s:
            # Truncate target solution to match source dim
            p1 = p_tensor[:, :D_s]      # First D_s dimensions
            p2 = p_tensor[:, D_s:]      # Remaining dimensions
            
            # Reconstruct the truncated part
            mu, logvar = source_vae.encode(p1)
            z = source_vae.reparameterize(mu, logvar)
            x_rec = source_vae.decode(z)
            
            # Concatenate reconstructed part with preserved dimensions
            x_new = torch.cat((x_rec, p2), dim=1).detach().numpy().flatten()
            
        else: # D_t < D_s
            # 1. Generate random latent vector z -> decode to z_prime
            z_rand = torch.randn(1, source_vae.latent_dim)
            z_prime = source_vae.decode(z_rand)
            
            # 2. Get the "excess" dimensions from z_prime
            z_prime_excess = z_prime[:, D_t:] 
            
            # 3. Concatenate target solution p with excess dimensions
            p_prime = torch.cat((p_tensor, z_prime_excess), dim=1)
            
            # 4. Reconstruct this combined vector using Source VAE
            mu, logvar = source_vae.encode(p_prime)
            z = source_vae.reparameterize(mu, logvar)
            q = source_vae.decode(z)
            
            # 5. Truncate back to target dimension D_t
            x_new = q[:, :D_t].detach().numpy().flatten()
            
        return x_new

    def generator_transfer(self, target_archive, source_vae, D_t, D_s):
        """
        Algorithm 5: Generator-based Method (GNT)[cite: 458].
        """
        # Sample from latent distribution N(0,1) [cite: 463]
        z = torch.randn(1, source_vae.latent_dim)
        
        # Logic for dimension mismatch [cite: 483-489]
        if D_t == D_s:
            # Simple generation
            x_new = source_vae.decode(z).detach().numpy().flatten()
            
        elif D_t > D_s:
            # Need to fill extra dimensions from a target solution
            if not target_archive: return None
            p = random.choice(list(target_archive.values()))
            p_tensor = torch.FloatTensor(p).unsqueeze(0)
            
            # Decode z to get base structure
            z_prime = source_vae.decode(z)
            
            # Get excess dimensions from target solution
            p1_excess = p_tensor[:, D_s:]
            
            # Concatenate
            x_new = torch.cat((z_prime, p1_excess), dim=1).detach().numpy().flatten()
            
        else: # D_t < D_s
            # Decode z and truncate
            z_prime = source_vae.decode(z)
            x_new = z_prime[:, :D_t].detach().numpy().flatten()
            
        return x_new

    def perform_transfer(self, target_archive, source_vae, D_t, D_s, batch_size=250):
        """
        Algorithm 3: ARAKT main loop[cite: 355].
        target_archive: Dict mapping behavior descriptor -> genotype (numpy array)
        """
        new_solutions = []
        
        # Determine method via Bandit
        method = self.select_transfer_method()
        
        for _ in range(batch_size):
            if method == 1:
                sol = self.autoencoder_transfer(target_archive, source_vae, D_t, D_s)
            else:
                sol = self.generator_transfer(target_archive, source_vae, D_t, D_s)
            
            if sol is not None:
                new_solutions.append((method, sol))
                
        return new_solutions

    def update_bandit(self, results):
        """
        Feedback loop for UCB1.
        results: List of (method, reward) tuples from the evaluation phase.
        """
        for method, reward in results:
            self.update_stats(method, reward)

    def receive_knowledge(self, target_worker, source_vae, source_dim, target_archive_mock, batch_size):
        """
        Orchestrates the reception of knowledge from a foreign VAE into the target worker.
        """
        if source_vae is None:
            return
            
        # 1. Generate new candidate solutions (raw vectors)
        new_solution_data = self.perform_transfer(
            target_archive=target_archive_mock,
            source_vae=source_vae,
            D_t=target_worker.task_dim,
            D_s=source_dim,
            batch_size=batch_size
        )

        feedback_results = []
        successful_transfers = 0

        # 2. Evaluate and inject into the target's MAP-Elites Archive
        for method_idx, new_vec in new_solution_data:
            # Use the target worker's domain knowledge to decode and evaluate
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
        
        # 3. Update the Bandit statistics based on how well the transfer performed
        self.update_bandit(feedback_results)
        
        task_identifier = getattr(target_worker, 'task_id', 'Unknown')
        print(f"Task {task_identifier} received transfer. Injected/Improved {successful_transfers}. Bandit Stats: {self.selected}")