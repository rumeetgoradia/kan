from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class ChebyshevKAN(nn.Module):
    def __init__(self, layers_hidden, polynomial_order=3, base_activation=nn.SiLU):
        super(ChebyshevKAN, self).__init__()

        self.layers_hidden = layers_hidden
        self.polynomial_order = polynomial_order
        self.base_activation = base_activation()

        self.base_weights = nn.ParameterList()
        self.poly_weights = nn.ParameterList()
        self.layer_norms = nn.ModuleList()

        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            self.base_weights.append(nn.Parameter(torch.randn(out_features, in_features)))
            self.poly_weights.append(nn.Parameter(torch.randn(out_features, in_features * (polynomial_order + 1))))
            self.layer_norms.append(nn.LayerNorm(out_features))

        for weight in self.base_weights:
            nn.init.xavier_uniform_(weight)
        for weight in self.poly_weights:
            nn.init.xavier_uniform_(weight)

    @lru_cache(maxsize=128)
    def compute_chebyshev_polynomials(self, x, order):
        T0 = x.new_ones(x.shape)
        if order == 0:
            return T0.unsqueeze(-1)
        T1 = x
        chebyshev_polys = [T0, T1]
        for _ in range(1, order):
            Tn = 2 * x * chebyshev_polys[-1] - chebyshev_polys[-2]
            chebyshev_polys.append(Tn)
        return torch.stack(chebyshev_polys, dim=-1)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.size()
        x = x.view(batch_size * seq_len, feature_dim)

        x = x.to(self.base_weights[0].device)

        for i, (base_weight, poly_weight, layer_norm) in enumerate(
                zip(self.base_weights, self.poly_weights, self.layer_norms)):
            base_output = F.linear(self.base_activation(x), base_weight)

            x_normalized = 2 * (x - x.min(dim=1, keepdim=True)[0]) / (
                    x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0] + 1e-8) - 1
            chebyshev_basis = self.compute_chebyshev_polynomials(x_normalized, self.polynomial_order)
            chebyshev_basis = chebyshev_basis.view(batch_size * seq_len, -1)

            poly_output = F.linear(chebyshev_basis, poly_weight)
            x = self.base_activation(layer_norm(base_output + poly_output))

        x = x.view(batch_size, seq_len, -1)
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
    model = ChebyshevKAN(layers_hidden=[43, 64, 1]).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping parameters
    patience = 3
    best_val_loss = float('inf')
    epochs_no_improve = 0

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
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                val_outputs = model(val_data)
                val_loss += criterion(val_outputs, val_labels).item()

                # Collect predictions and true labels
                preds = torch.argmax(val_outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())

        # Calculate average validation loss
        val_loss /= len(val_loader)

        # Calculate forecasting metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        mae = np.mean(np.abs(all_preds - all_labels))
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        mape = np.mean(np.abs((all_labels - all_preds) / all_labels)) * 100

        # Log metrics to a file
        with open('training_metrics.txt', 'a') as f:
            f.write(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, '
                    f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%\n')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Save the final model and optimizer state
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, 'model_checkpoint.pth')


if __name__ == '__main__':
    X_train = np.load('../data/prepared/sequenced/X_train.npy').astype(np.float32)
    y_train = np.load('../data/prepared/sequenced/y_train.npy').astype(np.float32)
    train(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
