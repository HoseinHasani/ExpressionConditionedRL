import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

output_dir = "data"
env_inference = "GoalReacher_sr"  
data_dir = os.path.join(output_dir, env_inference)
batch_size = 32
epochs = 50
learning_rate = 0.001
test_split_ratio = 0.2

conditions_path = os.path.join(data_dir, "conditions.npy")
labels_path = os.path.join(data_dir, "labels.npy")

conditions = np.load(conditions_path)
labels = np.load(labels_path)

conditions_tensor = torch.tensor(conditions, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

full_dataset = TensorDataset(conditions_tensor, labels_tensor)
test_size = int(len(full_dataset) * test_split_ratio)
train_size = len(full_dataset) - test_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

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

input_size = conditions.shape[1]
hidden_size = 128
output_size = len(np.unique(labels))

model = MLPClassifier(input_size, hidden_size, output_size)
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
