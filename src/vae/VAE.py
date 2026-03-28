import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# FILE VAE cũ, hiện tại đã dùng sang VAE2.py

class VAE(nn.Module):
    # latent_dim = 6 is in the paper
    def __init__(self, task_dim, latent_dim=6):
        """
        VAE Architecture.
        Args:
            task_dim (int): Dimension of the task (D).
            latent_dim (int): Dimension of the latent space (d_lat). Default is 6.
        """
        super(VAE, self).__init__()
        
        self.task_dim = task_dim
        self.latent_dim = latent_dim
        
        # Hidden layer dimension is 2 * d_lat
        hidden_dim = 2 * latent_dim

        # Encoder
        self.fc1 = nn.Linear(task_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, task_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss Function dựa trên Eq. (4)
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # MSE Loss
    MSE = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + beta * KLD

def train_vae(model, elite_genotypes, epochs=5, batch_size=32, learning_rate=1e-3):
    """
    Trains the VAE model on the genotypes of elite solutions.
    
    Args:
        model (VAE): The VAE instance to train.
        elite_genotypes (np.array or torch.Tensor): The dataset of elite solution genotypes (G(E_i)).
                                                    Shape should be (num_samples, task_dim).
                                                    NOTE: Data should ideally be normalized to [-1, 1] 
                                                    because the VAE output uses Tanh[cite: 324].
        epochs (int): Number of training epochs. Default is 5.
        batch_size (int): Size of training batches.
        learning_rate (float): Learning rate for Adam optimizer.
        
    Returns:
        model: The trained VAE model.
    """
    # Ensure model is in training mode
    model.train()
    
    # define optimizer: Paper specifies Adam optimizer 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Prepare data
    if not isinstance(elite_genotypes, torch.Tensor):
        # Convert numpy array to float tensor
        tensor_x = torch.FloatTensor(elite_genotypes)
    else:
        tensor_x = elite_genotypes.float()
        
    # Create dataloader for batching
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training Loop
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data,) in enumerate(dataloader):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass: obtain reconstruction, mu, and logvar [cite: 331]
            recon_batch, mu, logvar = model(data)
            
            # Calculate loss: MSE + KL Divergence [cite: 332]
            loss = loss_function(recon_batch, data, mu, logvar)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Optional: Print average loss per epoch
        avg_loss = total_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

    return model