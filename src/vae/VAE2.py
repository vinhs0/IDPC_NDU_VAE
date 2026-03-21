import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data

# Make sure your embedder is imported
from vae.Embedder import GraphEmbedder

class GraphVAE(nn.Module):
    def __init__(self, num_total_domains, num_nodes, latent_dim=6, n2v_embedding_dim=16):
        """
        Graph VAE Architecture using continuous Node2Vec structural embeddings.
        Args:
            num_total_domains (int): Total number of unique domains to embed (for classification).
            num_nodes (int): Total number of nodes in the graph (Task dimension).
            latent_dim (int): Dimension of the latent space (d_lat). Default is 6.
            n2v_embedding_dim (int): Dimension of the Node2Vec features from GraphEmbedder.
        """
        super(GraphVAE, self).__init__()
        
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.num_total_domains = num_total_domains
        
        # Features = Depth + n2v_embedding_dim (Topology)
        self.input_feature_dim = 1 + n2v_embedding_dim
        
        # Hidden layer dimension is 2 * d_lat
        hidden_dim = 2 * latent_dim

        # 1. Graph Encoder 
        self.conv1 = GCNConv(self.input_feature_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)

        # 2. MLP Decoder (Base)
        flattened_latent = latent_dim * num_nodes
        self.fc_decode = nn.Linear(flattened_latent, hidden_dim * num_nodes)
        
        # --- FIXED: Multi-Task Output Branches ---
        # Branch A: Predicts the continuous Depth
        self.fc_depth = nn.Linear(hidden_dim, 1)
        # Branch B: Predicts the categorical Node ID (Logits for CrossEntropy)
        self.fc_node_id = nn.Linear(hidden_dim, num_total_domains + 1)

    def prepare_graph_data(individuals) -> list:
        """
        Takes a list of Individuals, passes them through GraphEmbedder, 
        and extracts pure continuous inputs and separate targets for the VAE.
        """
        graph_data_list = []
        embed = GraphEmbedder()
        for ind in individuals:
            # 1. Get raw combined features from embedder: [NodeID, Depth, N2V_1, ... N2V_64]
            combined_features, edge_index = embed.get_embeddings(ind.get_chromosome())

            # 2. Split them up
            target_node_ids = combined_features[:, 0].long() # Keep for loss function
            depths = combined_features[:, 1].float().view(-1, 1)
            n2v_features = combined_features[:, 2:].float()

            # 3. Normalize Depth to [-1, 1] to match Tanh output
            max_depth = torch.max(depths).item() if len(depths) > 0 else 1.0
            if max_depth > 0:
                depths_norm = 2 * (depths / max_depth) - 1
            else:
                depths_norm = depths

            # 4. Create the pristine, continuous input X
            x_input = torch.cat([depths_norm, n2v_features], dim=-1)

            # 5. Build PyG Data object
            graph_data_list.append(Data(
                x=x_input, 
                edge_index=edge_index,
                target_node_ids=target_node_ids, 
                target_depths=depths_norm,
                max_depth_val=torch.tensor([max_depth], dtype=torch.float)
            ))
            
        return graph_data_list

    def encode(self, x, edge_index):
        """Encodes node features and graph structure to latent parameters."""
        h1 = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(h1, edge_index), self.conv_logvar(h1, edge_index)

    def reparameterize(self, mu, logvar):
        """Standard VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, batch_size):
        """
        Decodes latent vectors back to the Node features space.
        Splits into continuous Depth (Tanh) and categorical Node ID (Logits).
        """
        # Flatten and pass through base decoder
        z_flat = z.view(batch_size, self.num_nodes * self.latent_dim)
        h = F.relu(self.fc_decode(z_flat))
        
        # Reshape back to individual nodes: [total_nodes_in_batch, hidden_dim]
        h = h.view(batch_size * self.num_nodes, -1)
        
        # Branch A: Depth (Normalized -1 to 1)
        recon_depths = torch.tanh(self.fc_depth(h))
        
        # Branch B: Node ID (Raw logits for CrossEntropy)
        recon_logits = self.fc_node_id(h)
        
        return recon_depths, recon_logits

    def forward(self, x, edge_index, batch):
        batch_size = batch.max().item() + 1 if batch is not None else 1
        
        # Encode -> Reparameterize -> Decode
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        recon_depths, recon_logits = self.decode(z, batch_size)
        
        return recon_depths, recon_logits, mu, logvar

    def get_discrete_node_features(self, recon_depths, recon_logits, max_depth_val):
        """
        Helper method for Knowledge Transfer: 
        Converts VAE output back to discrete [Node_ID, Depth].
        """
        # 1. Decode Node ID: Take the class with the highest probability (argmax)
        discrete_node_ids = torch.argmax(recon_logits, dim=1)
        
        # 2. Decode Depth: Reverse the [-1, 1] normalization and round to nearest integer
        discrete_depths = torch.round(((recon_depths.view(-1) + 1) / 2) * max_depth_val).long()
        discrete_depths = torch.clamp(discrete_depths, min=0) # Prevent negative depth
        
        return discrete_node_ids, discrete_depths


# --- UPDATED LOSS FUNCTION ---
def loss_function(recon_depths, target_depths, recon_logits, target_node_ids, mu, logvar, beta=1.0):
    # 1. Depth Loss (Regression / Mean Squared Error)
    loss_depth = F.mse_loss(recon_depths, target_depths, reduction='mean')
    
    # 2. Node ID Loss (Classification / Cross Entropy)
    loss_node = F.cross_entropy(recon_logits, target_node_ids, reduction='mean')
    
    # 3. KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / recon_depths.size(0)
    
    # Total Loss combines all three
    return loss_depth + loss_node + beta * KLD


# --- UPDATED TRAINING LOOP ---
def train_vae(model, graph_data_list, epochs=5, batch_size=32, learning_rate=1e-3):
    """Trains the GraphVAE model on the graph structures of elite solutions."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=True)
    
    device = next(model.parameters()).device

    for epoch in range(epochs):
        total_loss = 0
        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # --- Pass the clean continuous 'x' to the forward pass ---
            recon_depths, recon_logits, mu, logvar = model(
                batch_data.x, 
                batch_data.edge_index, 
                batch_data.batch
            )
            
            # --- Calculate combined loss using separate targets ---
            loss = loss_function(
                recon_depths, batch_data.target_depths, 
                recon_logits, batch_data.target_node_ids, 
                mu, logvar
            )
    
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_data.num_graphs
            
        avg_loss = total_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    return model