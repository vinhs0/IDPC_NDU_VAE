import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    # latent_dim = 6 is in the paper
    def __init__(self, task_dim, latent_dim=6):
        """
        VAE Architecture based on Fig. 3 of the paper.
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
        """
        Encodes input to latent distribution parameters.
        """
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        """
        Standard VAE reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decodes latent vector back to task space.
        Uses Tanh activation for output as per Fig 3.
        """
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss Function based on Eq. (4)
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # MSE Loss
    MSE = F.mse_loss(recon_x, x, reduction='mean') # Paper defines MSE as mean
    
    # KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Adjust KLD by latent_dim if needed to match the scale of the paper's summation
    
    return MSE + beta * KLD