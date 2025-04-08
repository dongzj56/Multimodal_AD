import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import time
import logging
import os
import pandas as pd
from datasets.ADNI import ADNI, ADNI_transform
from models.Resnet3D import ResNet3D
from monai.data import Dataset
from tqdm import tqdm

# Configuration of logging
logging.basicConfig(format='[%(asctime)s]  %(message)s',
                    datefmt='%d.%m %H:%M:%S',
                    level=logging.INFO)


def print_dataset_info(dataset, name):
    """Print dataset statistics"""
    labels = [int(item['label']) for item in dataset]
    class_counts = np.bincount(labels)
    logging.info(f"{name} | Total: {len(dataset)} | Class 0: {class_counts[0]} | Class 1: {class_counts[1]}")


def save_checkpoint(state, filename):
    """Save model checkpoint"""
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")


def train_model(task='ADCN', num_epochs=50, batch_size=4, lr=1e-3, checkpoint_dir='./checkpoints',
                log_file='training_log1.csv'):
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet3D(in_channels=2, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load dataset
    dataroot = rf'adni_dataset'
    dataset = ADNI(label_file=f'{dataroot}/ADNI.csv', mri_dir=f'{dataroot}/MRI', pet_dir=f'{dataroot}/PET', task=task,
                   augment=True).data_dict

    # Split dataset
    train_data, temp_data = train_test_split(dataset, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Apply transformations
    train_transforms, val_transforms = ADNI_transform()
    train_dataset = Dataset(data=train_data, transform=train_transforms)
    val_dataset = Dataset(data=val_data, transform=val_transforms)
    test_dataset = Dataset(data=test_data, transform=val_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Create a directory for saving checkpoints if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Initialize a DataFrame to log results
    log_columns = ['epoch', 'train_loss', 'val_loss', 'val_auc', 'val_acc']
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame(columns=log_columns)

    best_val_acc = 0
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        # Training phase
        for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100):
            mri = data['MRI']
            pet = data['PET']
            labels = data['label']

            inputs = torch.cat([mri.unsqueeze(1), pet.unsqueeze(1)], dim=1).float().to(device)
            inputs = inputs.squeeze(2)
            targets = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation", ncols=100):
                mri = data['MRI']
                pet = data['PET']
                labels = data['label']

                inputs = torch.cat([mri.unsqueeze(1), pet.unsqueeze(1)], dim=1).float().to(device)
                inputs = inputs.squeeze(2)
                targets = labels.long().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        # Calculate metrics
        train_loss = epoch_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(all_labels, all_preds)
        val_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        # Print log
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch + 1}/{num_epochs} | Time: {epoch_time:.1f}s | "
                     f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                     f"AUC: {val_auc:.4f} | Acc: {val_acc:.4f}")

        # Save the results into the log DataFrame using pd.concat
        log_df = pd.concat([log_df, pd.DataFrame([{
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'val_acc': val_acc
        }])], ignore_index=True)

        # Save log to CSV
        log_df.to_csv(log_file, index=False)

        # Save checkpoint and the best model
        checkpoint_filename = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}_checkpoint.pth')
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, checkpoint_filename)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_model_filename = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(best_model_state, best_model_filename)
            logging.info(f"New best model saved with accuracy: {best_val_acc:.4f}")

    logging.info("Training complete!")

    # Load the best model and evaluate on the test dataset
    logging.info("Loading the best model for testing...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth')))
    model.eval()

    # Test phase
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", ncols=100):
            mri = data['MRI']
            pet = data['PET']
            labels = data['label']

            inputs = torch.cat([mri.unsqueeze(1), pet.unsqueeze(1)], dim=1).float().to(device)
            inputs = inputs.squeeze(2)
            targets = labels.long().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # Calculate metrics for the test set
    test_loss = test_loss / len(test_loader)
    test_auc = roc_auc_score(all_labels, all_preds)
    test_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

    logging.info(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")



if __name__ == "__main__":
    train_model(task='ADCN', num_epochs=50, batch_size=4, lr=1e-3, checkpoint_dir='./checkpoints',
                log_file='training_log1.csv')
