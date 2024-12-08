import torch
import torchvision
import torchvision.transforms as transforms
from baseline_res50_data import load_datasets
import os
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 
summary_writer = SummaryWriter('tensorboard')
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
num_epochs = 20
batch_size = 32
learning_rate = 1e-5


train_loader = load_datasets('train', batch_size)

# Load the ResNet50 model
model = torchvision.models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(in_features=2048, out_features=600, bias=True)

# Set the model to run on the device
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batches = len(train_loader)
# Train the model...
for epoch in range(num_epochs):
    batch_idx = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        summary_writer.add_scalar('loss_batch', loss.item(), epoch * total_batches + batch_idx)
        batch_idx += 1
    # Print the loss for every epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
# Save Model
torch.save(model.state_dict(), "./models/resnet50_hico_det.pth")

print("Model saved as resnet50_hico_det.pth")

print(f'Finished Training, Loss: {loss.item():.4f}')