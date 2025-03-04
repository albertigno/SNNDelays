import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Streamlit app
st.title("Live Training Loss Visualization")
st.write("This app visualizes the training loss of a simple PyTorch model in real-time.")

# Initialize model, loss, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create a placeholder for the live plot
plot_placeholder = st.empty()

# Generate some synthetic data
x_train = torch.arange(0, 10, 0.1).unsqueeze(1)
y_train = 2 * x_train + 3 + torch.randn(x_train.size()) * 0.5  # y = 2x + 3 + noise

# Training loop
num_epochs = 1000
loss_history = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss_history.append(loss.item())

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update the live plot every epoch
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        fig, ax = plt.subplots()
        ax.plot(loss_history, label="Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plot_placeholder.pyplot(fig)

    # Print the loss in the app
    st.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Final message
st.write("Training complete!")