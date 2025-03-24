import torch
import json
import pandas as pd
from model4 import EfficientCNN, get_dataset_and_samplers  # Import model & dataset loaders

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Function to evaluate the model
def evaluate_model(model, data_loader):
    """Evaluates the model accuracy and loss on the test set."""
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)

    return {"accuracy": accuracy, "average_loss": avg_loss}

# ✅ Main execution block
if __name__ == '__main__':
    print("🚀 Starting model evaluation...")

    # ✅ Get the correct number of classes from the dataset
    dataset, _, _, test_loader = get_dataset_and_samplers(batch_size=64, num_workers=0)
    K = len(dataset.class_to_idx)  # Automatically detect number of classes

    # ✅ Load the trained model
    model = EfficientCNN(K=K).to(device)
    model.load_state_dict(torch.load("plant_disease_model_1_state_dict.pt", map_location=device))
    model.eval()

    # ✅ Compute evaluation metrics
    metrics = evaluate_model(model, test_loader)

    # ✅ Save metrics as JSON
    with open("evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("📊 Evaluation results saved to `evaluation_results.json`.")
    print(f"✅ Model Accuracy: {metrics['accuracy']:.2f}%")
    print(f"📉 Average Loss: {metrics['average_loss']:.4f}")
