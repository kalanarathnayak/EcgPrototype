import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from config import *
from data_processing import preprocess_data, prepare_data
from model import TransformerECG, ECGDataset
from utils import (
    train_model, evaluate_model, plot_training_history,
    plot_roc_curve, plot_confusion_matrix
)

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Data preprocessing
    print("Starting data preprocessing...")
    beats, labels, patient_ids = preprocess_data(DATA_PATH, PATIENTS)

    # Prepare data splits
    print("\nPreparing data splits...")
    train_df, val_df, test_df = prepare_data(beats, labels, patient_ids)

    # Create datasets
    train_data = train_df.drop(['target', 'patient_id'], axis=1).values
    val_data = val_df.drop(['target', 'patient_id'], axis=1).values
    test_data = test_df.drop(['target', 'patient_id'], axis=1).values

    train_dataset = ECGDataset(train_data, augment=True)
    val_dataset = ECGDataset(val_data, augment=False)
    test_dataset = ECGDataset(test_data, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    print("\nInitializing model...")
    model = TransformerECG(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        dropout=DROPOUT
    )

    # Train model
    print("\nStarting model training...")
    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(train_losses, val_losses)

    # Compute reconstruction error threshold using validation data
    print("\nComputing threshold...")
    val_errors = []
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            error = torch.mean(torch.abs(output - batch), dim=1)
            val_errors.extend(error.cpu().numpy())
    val_errors = np.array(val_errors)
    threshold = np.percentile(val_errors, 87)
    print(f"Using threshold from 87th percentile: {threshold:.6f}")

    # Evaluate model on test set
    print("\nEvaluating model on test data...")
    predictions, errors, true_labels = evaluate_model(model, test_loader, test_df, threshold=threshold)

    # Calculate and print metrics
    f1 = f1_score(true_labels, predictions)
    print(f"\nF1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))

    # Plot confusion matrix and ROC curve
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(true_labels, predictions)
    print("\nPlotting ROC curve...")
    roc_auc = plot_roc_curve(true_labels, errors)
    print(f"ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    main()