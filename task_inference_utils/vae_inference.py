import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from task_inference_utils.base_inference import BaseTaskInference

class VAEInference(BaseTaskInference, nn.Module):
    def __init__(self, context_size=10, n_steps=4, state_dim=3, action_dim=2, hidden_dim=128, lr=1e-3):
        super().__init__(context_size)
        nn.Module.__init__(self) 
        self.n_steps = n_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        input_size = n_steps * (state_dim + action_dim + 1)


        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
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
        rewards = np.array(rewards).reshape(-1, 1)
        
        input_data = np.concatenate([states, actions, rewards], axis=-1)  # Shape will be (n_steps, state_dim + action_dim + 1)
        
        input_data = torch.tensor(input_data, dtype=torch.float32).view(1, -1)
        
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
            x = np.concatenate([states, actions, rewards], axis=-1).flatten()  # Flatten the data
            y = np.concatenate([trajectory_buffer[i+self.n_steps][0], [trajectory_buffer[i+self.n_steps][2]]])
            x_data.append(x)
            y_data.append(y)
        
        x_tensor = torch.tensor(np.array(x_data), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_data), dtype=torch.float32)
        return torch.utils.data.TensorDataset(x_tensor, y_tensor)

if __name__ == "__main__":
    def generate_dummy_data(n_samples=100, n_steps=4, state_dim=3, action_dim=2):
        trajectory_buffer = []
        for _ in range(n_samples):
            states = np.random.randn(n_steps + 1, state_dim)
            actions = np.random.randn(n_steps, action_dim)
            rewards = np.random.randn(n_steps)
            for i in range(n_steps):
                trajectory_buffer.append((states[i], actions[i], rewards[i]))
        return trajectory_buffer

    trajectory_buffer = generate_dummy_data(n_samples=200)

    vae_inference = VAEInference(context_size=10, n_steps=4, state_dim=3, action_dim=2, hidden_dim=128, lr=1e-3)

    latent = vae_inference.infer_task(trajectory_buffer[:4])
    print("Predicted latent vector:", latent)

    dataset = vae_inference.create_shuffled_dataset(trajectory_buffer)
    vae_inference.train_vae(dataset, epochs=2, batch_size=16)

    print("Minimal VAE test completed successfully!")
