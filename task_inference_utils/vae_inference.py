import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from task_inference_utils.base_inference import BaseTaskInference

class VAEInference(BaseTaskInference):
    def __init__(self, context_size=10, n_steps=4, hidden_dim=128, lr=1e-3):
        super().__init__(context_size)
        self.n_steps = n_steps
        self.encoder = nn.Sequential(
            nn.Linear(n_steps * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_size * 2)  
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(context_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) 
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def infer_task(self, trajectory_buffer):
        if len(trajectory_buffer) < self.n_steps:
            return np.zeros(self.context_size, dtype=np.float32)
        
        sample = trajectory_buffer[:self.n_steps]
        states, actions, rewards = zip(*sample)
        input_data = torch.tensor(np.concatenate([states, actions, rewards], axis=-1), dtype=torch.float32).view(1, -1)
        
        encoded = self.encoder(input_data)
        mu, log_var = encoded.chunk(2, dim=-1)
        latent = self.reparameterize(mu, log_var).detach().numpy().squeeze()
        
        return latent.astype(np.float32)
    
    def train_vae(self, dataset, epochs=100, batch_size=32):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            total_loss = 0
            for x, y in data_loader:
                self.optimizer.zero_grad()
                encoded = self.encoder(x)
                mu, log_var = encoded.chunk(2, dim=-1)
                latent = self.reparameterize(mu, log_var)
                decoded = self.decoder(latent)
                
                recon_loss = self.loss_fn(decoded, y)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + 0.1 * kl_loss
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    def create_shuffled_dataset(self, trajectory_buffer):
        x_data, y_data = [], []
        for i in range(len(trajectory_buffer) - self.n_steps):
            sample = trajectory_buffer[i:i+self.n_steps]
            states, actions, rewards = zip(*sample)
            x = np.concatenate([states, actions, rewards], axis=-1).flatten()
            y = np.concatenate([trajectory_buffer[i+self.n_steps][0], [trajectory_buffer[i+self.n_steps][2]]])
            x_data.append(x)
            y_data.append(y)
        
        x_tensor = torch.tensor(np.array(x_data), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_data), dtype=torch.float32)
        return torch.utils.data.TensorDataset(x_tensor, y_tensor)
