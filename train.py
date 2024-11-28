import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


class KANTrainer:
    def __init__(self, model_class, model_params, device, train_loader, val_loader, optimizer, criterion):
        self.model = model_class(**model_params).to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self):
        self.model.train()
        total_loss, total_accuracy = 0, 0
        all_preds, all_labels = [], []
        for sequences, labels in self.train_loader:
            sequences, labels = sequences.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(sequences).squeeze()  # Ensure output is the correct shape
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            preds = output.argmax(dim=1) if output.ndim > 1 else (
                        output > 0.5).float()  # Adjust for binary classification
            accuracy = (preds == labels).float().mean().item()
            total_loss += loss.item()
            total_accuracy += accuracy
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        return total_loss / len(self.train_loader), total_accuracy / len(self.train_loader), precision, recall

    def validate_epoch(self):
        self.model.eval()
        val_loss, val_accuracy = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for sequences, labels in self.val_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                output = self.model(sequences).squeeze()  # Ensure output is the correct shape
                val_loss += self.criterion(output, labels).item()
                preds = output.argmax(dim=1) if output.ndim > 1 else (
                            output > 0.5).float()  # Adjust for binary classification
                val_accuracy += (preds == labels).float().mean().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        return val_loss / len(self.val_loader), val_accuracy / len(self.val_loader), precision, recall

    def fit(self, epochs):
        train_metrics, val_metrics = [], []
        start_time = time.time()
        for epoch in tqdm(range(epochs), desc="Epoch Progress"):
            train_loss, train_accuracy, train_precision, train_recall = self.train_epoch()
            val_loss, val_accuracy, val_precision, val_recall = self.validate_epoch()
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
            train_metrics.append((train_loss, train_accuracy, train_precision, train_recall))
            val_metrics.append((val_loss, val_accuracy, val_precision, val_recall))
        training_time = time.time() - start_time
        return train_metrics, val_metrics, training_time


def train_and_validate(data, labels, model_class, model_params, epochs=15, batch_size=64):
    # Split data into training and validation sets
    dataset = TensorDataset(data, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.AdamW(model_class(**model_params).parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss() if labels.max() <= 1 else nn.CrossEntropyLoss()  # Adjust based on task
    trainer = KANTrainer(model_class, model_params, device, train_loader, val_loader, optimizer, criterion)
    train_metrics, val_metrics, training_time = trainer.fit(epochs)
    # Save the trained model's state dictionary
    model_save_path = f"./models/{model_class.__name__}_model.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(trainer.model.state_dict(), model_save_path)
    # Save metrics to a .npy file
    np.save(f"./models/{model_class.__name__}_metrics.npy",
            {'train': train_metrics, 'val': val_metrics, 'time': training_time})
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Train Metrics: {train_metrics}")
    print(f"Validation Metrics: {val_metrics}")
