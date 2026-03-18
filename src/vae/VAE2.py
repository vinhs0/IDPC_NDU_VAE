import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data

class GraphVAE(nn.Module):
    def __init__(self, node_feature_dim, num_nodes, latent_dim=6):
        """
        Graph VAE Architecture.
        Args:
            node_feature_dim (int): Number of features per node (e.g., 2 for [Node ID, Depth]).
            num_nodes (int): Total number of nodes in the graph (Task dimension).
            latent_dim (int): Dimension of the latent space (d_lat). Default is 6 [cite: 324-326].
        """
        super(GraphVAE, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        
        # Hidden layer dimension is 2 * d_lat [cite: 324-326]
        hidden_dim = 2 * latent_dim

        # 1. Graph Encoder (Cái này dùng Graph Convolutional Neural Network)
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)

        # 2. MLP Decoder 
        # Flattens the node embeddings to generate a graph-level output,
        # mapping back to the continuous range representing Node IDs.
        flattened_latent = latent_dim * num_nodes
        self.fc3 = nn.Linear(flattened_latent, hidden_dim * num_nodes)
        self.fc4 = nn.Linear(hidden_dim * num_nodes, node_feature_dim * num_nodes)

    def prepare_graph_data(individuals) -> list:
        """
        Takes a list of Individual objects and converts them into 
        a list of PyTorch Geometric Data objects.
        """
        graph_data_list = []
        
        for ind in individuals:
            x, edge_index = ind.get_graph_data()
            
            # Normalize the node features before converting to tensor
            # The node ID is the first feature, we normalize it to [-1, 1] 
            # to match the Tanh output of the VAE decoder.
            max_node = ind.total_domain
            min_node = 1
            
            # If the individual somehow has no nodes, skip
            if len(x) == 0:
                continue

            # Normalize Node IDs
            if max_node - min_node > 0:
                x[:, 0] = 2 * ((x[:, 0] - min_node) / (max_node - min_node)) - 1
            
            # Normalize Depth (Optional, but good for neural networks)
            max_depth = np.max(x[:, 1]) if len(x) > 0 else 1
            if max_depth > 0:
                 x[:, 1] = 2 * (x[:, 1] / max_depth) - 1

            # Convert numpy arrays to Torch Tensors
            x_tensor = torch.tensor(x, dtype=torch.float)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
            
            graph_data_list.append(Data(x=x_tensor, edge_index=edge_index_tensor))
            
        return graph_data_list

    def encode(self, x, edge_index):
        """
        Encodes node features and graph structure to latent distribution parameters.
        """
        h1 = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(h1, edge_index), self.conv_logvar(h1, edge_index)

    def reparameterize(self, mu, logvar):
        """
        Standard VAE reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, batch_size):
        """
        Decodes latent vectors back to the node feature space.
        Uses Tanh activation to enforce range constraints[cite: 326].
        """
        # Flatten node-level embeddings into a graph-level embedding sequence
        # Shape: [batch_size * num_nodes, latent_dim] -> [batch_size, num_nodes * latent_dim]
        z_flat = z.view(batch_size, self.num_nodes * self.latent_dim)
        
        h3 = F.relu(self.fc3(z_flat))
        out = torch.tanh(self.fc4(h3))
        
        # Reshape back to [batch_size * num_nodes, node_feature_dim] to match input 'x'
        return out.view(batch_size * self.num_nodes, self.node_feature_dim)

    def forward(self, x, edge_index, batch):
        # 'batch' is a PyG assignment array mapping nodes to their respective graphs
        batch_size = batch.max().item() + 1 if batch is not None else 1
        
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, batch_size)
        
        return recon_x, mu, logvar


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Mean square error
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Average KLD across the total number of nodes in the batch 
    # to prevent it from overpowering the MSE reconstruction loss.
    # Đoạn này là mới có, cần xem
    total_nodes = x.size(0) 
    KLD = KLD / total_nodes
    
    # Theo công thức cũ (trong paper), total loss là tổng của 2 thành phần (KL Div và MSE)
    return MSE + beta * KLD


def train_vae(model, graph_data_list, epochs=10, batch_size=32, learning_rate=1e-3):
    """
    Trains the GraphVAE model on the graph structures of elite solutions.
    
    Args:
        model (GraphVAE): The GraphVAE instance to train.
        graph_data_list (List[Data]): A list of torch_geometric.data.Data objects 
                                      containing the 'x' and 'edge_index' of each elite.
        epochs (int): Number of training epochs. Default is 5.
        batch_size (int): Size of training batches.
        learning_rate (float): Learning rate for Adam optimizer[cite: 327].
        
    Returns:
        model: The trained GraphVAE model.
    """
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Use PyTorch Geometric's specialized DataLoader to handle graph batching
    dataloader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_data in dataloader:
            optimizer.zero_grad()
            
            # Forward pass using graph structure [cite: 331]
            recon_batch, mu, logvar = model(batch_data.x, batch_data.edge_index, batch_data.batch)
            
            # Calculate loss: MSE + KL Divergence [cite: 332]
            loss = loss_function(recon_batch, batch_data.x, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            # PyG batches graphs dynamically; num_graphs tracks how many are in this specific batch
            total_loss += loss.item() * batch_data.num_graphs
            
        # Uncomment to track training progression
        avg_loss = total_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    return model