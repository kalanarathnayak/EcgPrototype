import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.0005):
    """
    Train the model using:
      - AdamW optimizer with weight decay
      - OneCycleLR learning rate scheduling
      - Aggressive gradient clipping
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)

    criterion = nn.L1Loss()  # MAE loss for reconstruction
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1,
        div_factor=50.0
    )

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            optimizer.step()
            scheduler.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                epoch_val_loss += loss.item()
        epoch_val_loss /= len(val_loader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}: Training Loss: {epoch_train_loss:.6f} | Validation Loss: {epoch_val_loss:.6f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping after {epoch + 1} epochs')
                break

    return model, train_losses, val_losses

def evaluate_model(model, test_loader, test_df, threshold=None):
    """Evaluate the model using reconstruction error and a given threshold."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    reconstruction_errors = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            error = torch.mean(torch.abs(output - batch), dim=1)
            reconstruction_errors.extend(error.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)
    predictions = (reconstruction_errors > threshold).astype(int)
    predicted_labels = 1 - predictions
    true_labels = test_df['target'].apply(lambda x: 1 if x == 1 else 0).values

    return predicted_labels, reconstruction_errors, true_labels

def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc_curve(true_labels, errors):
    """Plot ROC curve and calculate AUC."""
    fpr, tpr, _ = roc_curve(true_labels, -errors)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return roc_auc

def plot_confusion_matrix(true_labels, predictions):
    """Plot confusion matrix."""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()