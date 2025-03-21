import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import timm
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import psutil
import subprocess
import time

# GPU Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resource usage functions
def log_resource_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0
    print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")
    print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} MB, GPU Memory Reserved: {gpu_memory_reserved:.2f} MB")

def get_gpu_power_usage():
    if torch.cuda.is_available():
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,nounits,noheader"])
        power_usage = float(result.decode().strip())
        print(f"GPU Power Usage: {power_usage:.2f} W")

# Dataset Path
from google.colab import drive
drive.mount('/content/drive')
dataset_path = '/content/drive/MyDrive/giik/Eyes'

# Data Preprocessing (Resizing and Normalization)
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize all images to 512x512
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Dataset
dataset = ImageFolder(dataset_path, transform=transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoader
batch_size = 16  # Batch size is kept small for memory management with 512x512 dimensions
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# To Check Classes
print(f"Classes: {dataset.classes}")

vit_model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    img_size=512,  # 512x512 input size
    num_classes=len(dataset.classes)
)
vit_model.to(device)

swin_model = timm.create_model(
    'swin_base_patch4_window7_224',
    pretrained=True,
    img_size=512,  # 512x512 input size
    num_classes=len(dataset.classes)
)
swin_model.to(device)

def train_and_evaluate_model_with_plots(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    model.to(device)

    # Lists to Save Metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []

    start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 30)

        # Resource usage report (Epoch start)
        log_resource_usage()
        if torch.cuda.is_available():
            get_gpu_power_usage()

        # Training Phase
        model.train()
        train_loss = 0
        all_train_labels = []
        all_train_preds = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect Loss and Predictions
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds)

        # Compute Training Metrics
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted')
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted')
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')

        # Save Training Metrics
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)

        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Class-Based Report
        print("\nTrain Classification Report:")
        print(classification_report(all_train_labels, all_train_preds, target_names=dataset.classes))

        # Validation Phase
        model.eval()
        val_loss = 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward Pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Collect Loss and Predictions
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds)

        # Compute Validation Metrics
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')

        # Save Validation Metrics
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Class-Based Report
        print("\nValidation Classification Report:")
        print(classification_report(all_val_labels, all_val_preds, target_names=dataset.classes))

    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")

    # Plotting Metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                 train_precisions, val_precisions, train_recalls, val_recalls,
                 train_f1s, val_f1s)

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                 train_precisions, val_precisions, train_recalls, val_recalls,
                 train_f1s, val_f1s):
    epochs = range(1, len(train_losses) + 1)

    # Loss Graph
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Accuracy Graph
    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Precision Graph
    plt.figure()
    plt.plot(epochs, train_precisions, label='Train Precision')
    plt.plot(epochs, val_precisions, label='Validation Precision')
    plt.title('Precision Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

    # Recall Graph
    plt.figure()
    plt.plot(epochs, train_recalls, label='Train Recall')
    plt.plot(epochs, val_recalls, label='Validation Recall')
    plt.title('Recall Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()

    # F1 Score Graph
    plt.figure()
    plt.plot(epochs, train_f1s, label='Train F1 Score')
    plt.plot(epochs, val_f1s, label='Validation F1 Score')
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.show()


