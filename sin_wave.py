import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# generate data
x = np.linspace(0, 500, 1000)
y = np.sin(2 * np.pi * x / 100) 

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Sine Wave")
plt.show()


# Prepare time series dataset (50 time steps -> next value)
seq_length = 50
X, Y = [], []
for i in range(len(y) - seq_length):
    X.append(y[i:i + seq_length])
    Y.append(y[i + seq_length])

X = np.array(X)
Y = np.array(Y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, seq_length, 1)  # Shape should be (batch_size, seq_length, input_size)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

print(f"Input shape: {X_tensor.shape}, Output shape: {Y_tensor.shape}")

# Split into training (90%) and validation (10%)
train_ratio = 0.9
train_size = int(train_ratio * len(X_tensor))

X_train, X_val = X_tensor[:train_size], X_tensor[train_size + seq_length:]
Y_train, Y_val = Y_tensor[:train_size], Y_tensor[train_size + seq_length:]

print(f"Train set size: {X_train.shape}, Validation set size: {X_val.shape}")


# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # Define the RNN layer
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer to output prediction

    def forward(self, x):
        # Forward pass through the RNN layer
        out, _ = self.rnn(x)  
        # We only care about the last output for the prediction
        out = out[:, -1, :]  
        out = self.fc(out)  # Pass the output through the fully connected layer
        return out

# Hyperparameters
input_size = 1   # Since we're predicting a single value at each time step
hidden_size = 64  # Size of the RNN's hidden state
output_size = 1   # We want to predict a single value
learning_rate = 0.001

model = RNNModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Print the model architecture
print(model)


# Training loop
num_epochs = 100
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    # Train the model
    model.train()
    optimizer.zero_grad()  # Zero the gradients
    output = model(X_train)  # Forward pass
    loss = criterion(output, Y_train)  # Compute the loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update the weights

    # Calculate validation loss
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, Y_val)

    # Store losses for plotting
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# Plot the training and validation loss
plt.plot(range(num_epochs), train_losses, label="Training Loss")
plt.plot(range(num_epochs), val_losses, label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(X_val).cpu().numpy()

# Plot predictions vs true values
plt.plot(Y_val, label='True Values')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.title('Predictions vs True Values')
plt.show()
