import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class WaveletKAN(nn.Module):
    def __init__(self, layers_hidden, wavelet_name='db1', base_activation=nn.SiLU):
        super(WaveletKAN, self).__init__()

        self.layers_hidden = layers_hidden
        self.wavelet_name = wavelet_name
        self.base_activation = base_activation()

        self.base_weights = nn.ParameterList()
        self.wavelet_weights = nn.ParameterList()
        self.layer_norms = nn.ModuleList()

        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            # Adjust wavelet weight initialization to match the doubled feature size
            self.wavelet_weights.append(
                nn.Parameter(torch.randn(out_features, in_features * 2)))  # Approximation and detail coefficients
            self.layer_norms.append(nn.LayerNorm(out_features))

        for weight in self.base_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.wavelet_weights:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

        # Add a final linear layer to map to the output size
        self.output_layer = nn.Linear(layers_hidden[-1], 1)

    def compute_wavelet_transform(self, x):
        wavelet_coeffs = []
        for i in range(x.size(0)):
            coeffs = [pywt.dwt(x[i, :, j].cpu().numpy(), self.wavelet_name) for j in range(x.size(2))]
            # Concatenate approximation and detail coefficients
            coeffs = [np.concatenate(c) for c in coeffs]
            wavelet_coeffs.append(np.concatenate(coeffs))  # Concatenate along the feature dimension

        return torch.tensor(wavelet_coeffs, dtype=torch.float32, device=x.device).view(x.size(0), -1)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.size()
        x = x.view(batch_size * seq_len, feature_dim)

        x = x.to(self.base_weights[0].device)

        for i, (base_weight, wavelet_weight, layer_norm) in enumerate(
                zip(self.base_weights, self.wavelet_weights, self.layer_norms)):
            base_output = F.linear(self.base_activation(x), base_weight)

            wavelet_basis = self.compute_wavelet_transform(x.view(batch_size, seq_len, feature_dim))
            wavelet_basis = wavelet_basis.view(batch_size * seq_len, -1)

            wavelet_output = F.linear(wavelet_basis, wavelet_weight)
            x = self.base_activation(layer_norm(base_output + wavelet_output))

        # Reshape back to original batch and sequence dimensions
        x = x.view(batch_size, seq_len, -1)

        # Aggregate sequence dimension (e.g., mean or sum) before final output
        x = x.mean(dim=1)  # or could use sum(dim=1)

        # Final output layer
        x = self.output_layer(x).squeeze(-1)
        return x


def train(train_data: torch.Tensor, train_labels: torch.Tensor, batch_size=64):
    # Check if CUDA is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=42
    )

    # Create DataLoader for training and validation
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = WaveletKAN(layers_hidden=[43, 64, 128, 64]).to(device)
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_data, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Calculate average training loss
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                val_outputs = model(val_data)
                val_loss += criterion(val_outputs, val_labels).item()

        # Calculate average validation loss
        val_loss /= len(val_loader)

        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')


if __name__ == '__main__':
    X_train = np.load('../data/prepared/sequenced/X_train.npy').astype(np.float32)
    y_train = np.load('../data/prepared/sequenced/y_train.npy').astype(np.float32)
    train(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
