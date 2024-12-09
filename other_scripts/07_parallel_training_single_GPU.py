import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import multiprocessing
import time

# Define a simple MNIST classifier
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

# Training function
def train_model(lr, run_id):
    print(f"Starting training for Run {run_id} with learning rate {lr}")

    # Load MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Model, loss, optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(2):  # Short epochs for demonstration
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Save the model
    torch.save(model.state_dict(), f"model_run_{run_id}.pt")
    print(f"Finished training for Run {run_id}")

# Main function to manage parallel processes
if __name__ == "__main__":
    # Define different learning rates for each process
    learning_rates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]

    start_time = time.time()

    # Create and start processes
    processes = []
    for i, lr in enumerate(learning_rates):
        process = multiprocessing.Process(target=train_model, args=(lr, i))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("All training runs completed!")

    end_time = time.time()
    print(f"Total training time for parallel runs: {end_time - start_time:.2f} seconds")
