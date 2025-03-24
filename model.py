import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from datetime import datetime
import os
import sys

# Define the model class outside the `if __name__ == "__main__"` block
class EfficientCNN(nn.Module):
    def __init__(self, K):
        super(EfficientCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 28 * 28, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        return x

# Define dataset loading function outside the main block
def get_dataset_and_samplers(batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder("\Dataset\Plant_leave_diseases_dataset_without_augmentation", transform=transform)
    indices = list(range(len(dataset)))
    split = int(np.floor(0.85 * len(dataset)))
    validation_split = int(np.floor(0.70 * split))
    np.random.shuffle(indices)

    train_indices, validation_indices, test_indices = (
        indices[:validation_split],
        indices[validation_split:split],
        indices[split:]
    )

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)

    return dataset, train_loader, validation_loader, test_loader

# Define checkpoint functions outside the main block
def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    try:
        checkpoint = torch.load(filename, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch
    except FileNotFoundError:
        print("No checkpoint found. Starting from scratch.")
        return 0

# Define training function outside the main block
def batch_gd(model, criterion, optimizer, train_loader, validation_loader, epochs, start_epoch=0, save_interval=1, checkpoint_filename="checkpoint.pth"):
    total_steps = len(train_loader) * epochs  # Total iterations
    step = 0  # Initialize step counter

    for e in range(start_epoch, epochs):
        t0 = datetime.now()
        model.train()
        train_loss = []
    
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # Update progress percentage
            step += 1
            progress = (step / total_steps) * 100
            sys.stdout.write(f"\rEpoch [{e+1}/{epochs}] - Batch [{batch_idx+1}/{len(train_loader)}] - Progress: {progress:.2f}%")
            sys.stdout.flush()

        train_loss = np.mean(train_loss)

        model.eval()
        validation_loss = []
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = criterion(output, targets)
                validation_loss.append(loss.item())
        validation_loss = np.mean(validation_loss)

        dt = datetime.now() - t0
        print(f"\nEpoch {e+1}/{epochs} | Train Loss: {train_loss:.3f} | Validation Loss: {validation_loss:.3f} | Duration: {dt}")

        if (e + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, e + 1, checkpoint_filename)

# Ensure training is only executed when running this script directly
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, train_loader, validation_loader, test_loader = get_dataset_and_samplers()

    model = EfficientCNN(len(dataset.class_to_idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Load previous training checkpoint (if exists)
    start_epoch = load_checkpoint(model, optimizer)
    
    # Train the model
    batch_gd(model, criterion, optimizer, train_loader, validation_loader, epochs=3, start_epoch=start_epoch)

    # Save the trained model
    torch.save(model, 'plant_disease_model_1.pt')
    torch.save(model.state_dict(), 'plant_disease_model_1_state_dict.pt')
