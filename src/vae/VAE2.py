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
    def __init__(self, num_total_domains, num_nodes, latent_dim=8, n2v_embedding_dim=15):
        """
        Graph VAE Architecture using continuous Node2Vec structural embeddings.
        Args:
            num_total_domains (int): Total number of unique domains to embed (for classification).
            num_nodes (int): Total number of nodes in the graph (Task dimension).
            latent_dim (int): Dimension of the latent space (d_lat). Default is 8.
            n2v_embedding_dim (int): Dimension of the Node2Vec features from GraphEmbedder.
        """
        super(GraphVAE, self).__init__()
        
        self.num_nodes = num_nodes  
        self.latent_dim = latent_dim
        self.num_total_domains = num_total_domains
        
        # Features = Depth + n2v_embedding_dim (Topology)
        # chỗ này thử nghiệm dùng 15 n2v embeddings dim + 1 cái cho depth => tổng là 16
        self.input_feature_dim = n2v_embedding_dim + 1
        
        # Hidden layer dimension is 2 * d_lat
        hidden_dim = 2 * latent_dim

        # 1. Graph Encoder 
        self.conv1 = GCNConv(self.input_feature_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)

        # 2. MLP Decoder (Base)
        flattened_latent = latent_dim * num_nodes
        self.fc_decode = nn.Linear(flattened_latent, hidden_dim * num_nodes)
        
        # Dự đoán Depth
        self.fc_depth = nn.Linear(hidden_dim, 1)
        # Dự đoán xem Domain này là Domain nào (Domain ID)
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

            # Normalize depth để khớp với output mô hình dự đoán (theo hàm tanh, sẽ nằm trong khoảng [-1,1])
            max_depth = torch.max(depths).item() if len(depths) > 0 else 1.0
            if max_depth > 0:
                depths_norm = 2 * (depths / max_depth) - 1
            else:
                depths_norm = depths

            # Nối (concat) giữa độ sâu và node embedding features (16)
            x_input = torch.cat([depths_norm, n2v_features], dim=-1)

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

def loss_function(recon_depths, target_depths, recon_logits, target_node_ids, mu, logvar, beta=1.0):
    # loss của depth mô hình tiên đoán ra và depth thực tế của cái node đó
    loss_depth = F.mse_loss(recon_depths, target_depths, reduction='mean')
    # Domain ID (dùng để phân biệt điểm biên trong 1 domain)
    loss_node = F.cross_entropy(recon_logits, target_node_ids, reduction='mean')
    
    # KLD là loss khi VAE cố gắng học được latent representation của vector embeddings
    # Công thức này là công thức trong paper
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / recon_depths.size(0)
    
    total_loss = loss_depth + beta * KLD + loss_node
    return total_loss, loss_depth, loss_node, KLD

def train_vae(model, graph_data_list, epochs=5, batch_size=32, learning_rate=1e-3):
    """Trains the GraphVAE model on the graph structures of elite solutions."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=True)
    
    device = next(model.parameters()).device

    # Initialize the loss tracking dictionary
    loss_history = {'total': [], 'depth': [], 'node': [], 'kld': []}

    for epoch in range(epochs):
        # Accumulators for the different loss components
        total_loss_accum = 0.0
        depth_loss_accum = 0.0
        node_loss_accum = 0.0
        kld_accum = 0.0
        
        for batch_data in dataloader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # --- Pass the clean continuous 'x' to the forward pass ---
            recon_depths, recon_logits, mu, logvar = model(
                batch_data.x, 
                batch_data.edge_index, 
                batch_data.batch
            )
            
            # --- Unpack the 4 separate loss values ---
            loss, l_depth, l_node, l_kld = loss_function(
                recon_depths, batch_data.target_depths, 
                recon_logits, batch_data.target_node_ids, 
                mu, logvar
            )
    
            loss.backward()
            optimizer.step()
            
            # Multiply by num_graphs to get the raw sum for accurate averaging later
            num_graphs = batch_data.num_graphs
            total_loss_accum += loss.item() * num_graphs
            depth_loss_accum += l_depth.item() * num_graphs
            node_loss_accum += l_node.item() * num_graphs
            kld_accum += l_kld.item() * num_graphs
            
        # Calculate final averages for this epoch
        n = len(dataloader.dataset)
        avg_total = total_loss_accum / n
        avg_depth = depth_loss_accum / n
        avg_node = node_loss_accum / n
        avg_kld = kld_accum / n
        
        # Save the averages to the history dictionary
        loss_history['total'].append(avg_total)
        loss_history['depth'].append(avg_depth)
        loss_history['node'].append(avg_node)
        loss_history['kld'].append(avg_kld)
        
        # Luu y: ở đây Node ID là để VAE đoán xem domain này là domain nào 
        # (categorical), và đây là bắt buộc vì điểm biên được gán theo domain ID
        print(f"Epoch {epoch+1:03d}/{epochs:03d} | Total: {avg_total:.4f} -> [Node ID: {avg_node:.4f} | Depth MSE: {avg_depth:.4f} | KLD: {avg_kld:.4f}]", flush=True)

    # Return BOTH the model and the history to prevent the unpacking error
    return model, loss_history

# def train_vae(model, graph_data_list, epochs=5, batch_size=32, learning_rate=1e-3):
#     """Trains the GraphVAE model on the graph structures of elite solutions."""
#     model.train()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     dataloader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=True)
    
#     device = next(model.parameters()).device

#     for epoch in range(epochs):
#         # Accumulators for the different loss components
#         total_loss_accum = 0.0
#         depth_loss_accum = 0.0
#         node_loss_accum = 0.0
#         kld_accum = 0.0
        
#         for batch_data in dataloader:
#             batch_data = batch_data.to(device)
#             optimizer.zero_grad()
            
#             # --- Pass the clean continuous 'x' to the forward pass ---
#             recon_depths, recon_logits, mu, logvar = model(
#                 batch_data.x, 
#                 batch_data.edge_index, 
#                 batch_data.batch
#             )
            
#             # --- Unpack the 4 separate loss values ---
#             loss, l_depth, l_node, l_kld = loss_function(
#                 recon_depths, batch_data.target_depths, 
#                 recon_logits, batch_data.target_node_ids, 
#                 mu, logvar
#             )
    
#             loss.backward()
#             optimizer.step()
            
#             # Multiply by num_graphs to get the raw sum for accurate averaging later
#             num_graphs = batch_data.num_graphs
#             total_loss_accum += loss.item() * num_graphs
#             depth_loss_accum += l_depth.item() * num_graphs
#             node_loss_accum += l_node.item() * num_graphs
#             kld_accum += l_kld.item() * num_graphs
            
#         # Calculate final averages for this epoch
#         n = len(dataloader.dataset)
#         avg_total = total_loss_accum / n
#         avg_depth = depth_loss_accum / n
#         avg_node = node_loss_accum / n
#         avg_kld = kld_accum / n
        
#         # Luu y: ở đây Node ID là để VAE đoán xem domain này là domain nào 
#         # (categorical), và đây là bắt buộc vì điểm biên được gán theo domain ID
#         print(f"Epoch {epoch+1:03d}/{epochs:03d} | Total: {avg_total:.4f} -> [Node ID: {avg_node:.4f} | Depth MSE: {avg_depth:.4f} | KLD: {avg_kld:.4f}]", flush=True)

#     return model