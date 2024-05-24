import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Define the MLP model class
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_units, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

# Function to compute validation loss
def compute_validation_loss(model, loader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

# Function to train the model and return final losses
def train_model(hidden_units):
    model = MLP(input_dim=d, hidden_units=hidden_units, output_dim=1)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    val_loss = compute_validation_loss(model, val_loader)
    return train_loss, val_loss

if __name__ == '__main__':
    np.random.seed(42)
    n = 500
    d = 20
    k_values = [1, n//d, (n+100)//d, (n**2)//d, ((n+100)**2//d), ((n+200)**2)//d]

    X = np.random.randn(n, d)
    true_weights = np.random.randn(d, 1)
    y = X @ true_weights + np.random.randn(n, 1) * 0.5

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    train_losses = []
    val_losses = []
    param_counts = []

    for k in k_values:
        train_loss, val_loss = train_model(k)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        param_counts.append(k * d)

    plt.figure(figsize=(8, 6))
    plt.plot(param_counts, train_losses, marker='o', label='Training Loss')
    plt.plot(param_counts, val_losses, marker='o', label='Validation Loss')
    plt.xlabel('Number of Parameters (kd)')
    plt.ylabel('Loss')
    plt.title('Double Descent Phenomenon')
    plt.legend()
    plt.tight_layout()
    plt.savefig('experiments/results/figures/double_descent.png', dpi=300)  # Save as PNG file with high resolution

    plt.show()