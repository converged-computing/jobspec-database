# driver.py
import torch
from model import Simple3DCNN
# Assuming utils.py contains necessary data loading and preprocessing functions
from utils import load_data
import json

# Load parameters from JSON file
with open('params.json') as f:
    params = json.load(f)


# Hyperparameters (these should be tuned for your specific task)
# Extract parameters
batch_size = params['batch_size']
learning_rate = params['learning_rate']
num_epochs = params['num_epochs']
num_classes = params['num_classes']
save_path = params['data_paths']['save_path']

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
train_loader, validation_loader = load_data(batch_size=batch_size, data_paths=params['data_paths'])


# Initialize the model
model = Simple3DCNN(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Function to compute the accuracy
def compute_accuracy(loader, model):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Training loop (simplified, without actual data loading)
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):  # Should be defined properly
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    val_accuracy = compute_accuracy(validation_loader, model)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%')

    # Save the model if it has the best validation accuracy so far
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), save_path)
        print(f'Model saved to {save_path}')

print('Finished Training')

# If you want to save the final model regardless of its accuracy:
final_model_path = 'final_model.pth'
torch.save(model.state_dict(), final_model_path)
print(f'Final model saved to {final_model_path}')


# Code for validation and testing would go here

# Testing after training completion
test_accuracy = compute_accuracy(test_loader, model)
print(f'Test Accuracy: {test_accuracy:.2f}%')

print('Finished Training')