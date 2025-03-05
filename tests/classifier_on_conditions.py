import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from general_utils import fix_seed

fix_seed(seed=0)

output_dir = "data"
env_name = "Pendulum-v1"
inference_type = "sr"

env_inference =  f"{env_name}_{inference_type}"
data_dir = os.path.join(output_dir, env_inference)
batch_size = 32
epochs = 40
learning_rate = 0.001
test_split_ratio = 0.2
# n_tasks = 5
apply_normalization = True


config_path = os.path.join('../configs', f"{env_name}.json")
with open(config_path, 'r') as f:
    config = json.load(f)

n_tasks = config['n_tasks']


conditions_path = os.path.join(data_dir, "conditions.npy")
labels_path = os.path.join(data_dir, "labels.npy")

conditions = np.load(conditions_path)
labels = np.load(labels_path)
labels = labels % n_tasks

if apply_normalization:
    mean = conditions.mean(0)
    std = conditions.std(0)
    conditions = (conditions - mean) / (std + 1e-6)

conditions_tensor = torch.tensor(conditions, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

chunk_indices = []
start_idx = 0

for i in range(1, len(labels)):
    if labels[i] != labels[i - 1]:  
        chunk_indices.append((start_idx, i))  
        start_idx = i  
chunk_indices.append((start_idx, len(labels)))  

np.random.shuffle(chunk_indices)

test_chunk_count = int(len(chunk_indices) * test_split_ratio)
test_chunks = chunk_indices[:test_chunk_count]
train_chunks = chunk_indices[test_chunk_count:]

train_indices = [i for start, end in train_chunks for i in range(start, end)]
test_indices = [i for start, end in test_chunks for i in range(start, end)]

train_dataset = Subset(TensorDataset(conditions_tensor, labels_tensor), train_indices)
test_dataset = Subset(TensorDataset(conditions_tensor, labels_tensor), test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

model = MLPClassifier(conditions.shape[1], 128, n_tasks)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model():
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_conditions, batch_labels in train_loader:
            batch_conditions, batch_labels = batch_conditions.to(device), batch_labels.to(device)

            outputs = model(batch_conditions)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_conditions, batch_labels in test_loader:
            batch_conditions, batch_labels = batch_conditions.to(device), batch_labels.to(device)

            outputs = model(batch_conditions)
            _, predicted = torch.max(outputs, 1)

            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

train_model()
evaluate_model()
