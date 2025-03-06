import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt

# load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
testset = CIFAR10("./dataset", train=False, download=True, transform=transform)


NUM_CLIENTS = 10  
partition_size = len(trainset) // NUM_CLIENTS
lengths = [partition_size] * NUM_CLIENTS
generator = torch.Generator().manual_seed(42)  
datasets = random_split(trainset, lengths, generator=generator)

trainloaders, valloaders = [], []
for ds in datasets:
    len_val = int(len(ds) * 0.1)
    len_train = len(ds) - len_val
    ds_train, ds_val = random_split(ds, [len_train, len_val], generator=generator)

    trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
    valloaders.append(DataLoader(ds_val, batch_size=32, shuffle=False))

testloader = DataLoader(testset, batch_size=32, shuffle=False)

class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear (120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        # if verbose:
        #     print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

# trainloader = trainloaders[0]
# valloader = valloaders[0]
# net = Net().to(DEVICE)
# for epoch in range(5):
#     train(net, trainloader, 1)
#     loss, accuracy = test(net, valloader)
#     print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")
# loss, accuracy = test(net, testloader)
# print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")


# Initialize the global model
global_model = Net().to(DEVICE)

# Set number of rounds and local epochs
rounds = 5
local_epochs = 2

def fedavg(global_model, local_models, client_weights):
    
    # Get the state dict of the global model
    global_dict = global_model.state_dict()

    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]
    
    # Aggregate the weights from all clients
    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key])  # Initialize to zero
        
        for i, model in enumerate(local_models):
            local_dict = model.state_dict()
            global_dict[key] += normalized_weights[i] * local_dict[key]
    
    # Update the global model with the aggregated weights
    global_model.load_state_dict(global_dict)

# Initialize lists to store the results
test_losses, test_accuracies = [], []

for round_num in range(rounds):
    local_models, client_weights = [], []

    # Train local models on each client
    for i, trainloader in enumerate(trainloaders):
        local_model = copy.deepcopy(global_model)
        train(local_model, trainloader, epochs=local_epochs)
        
        local_models.append(local_model)
        client_weights.append(len(trainloader.dataset))  # Weight based on dataset size

    # Federated Averaging to update the global model
    fedavg(global_model, local_models, client_weights)

    # Evaluate the global model on the test set
    loss, accuracy = test(global_model, testloader)
    
    # Store the results
    test_losses.append(loss)
    test_accuracies.append(accuracy)

    print(f"Round {round_num + 1}, Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")


# Plot the results
rounds_range = range(1, rounds + 1)
plt.figure(figsize=(5, 4))
plt.plot(rounds_range, test_accuracies, label='Test Accuracy', color='green', marker='o')
plt.xlabel('Communication Round')
plt.ylabel('Test Accuracy')
plt.xticks(rounds_range)
plt.tight_layout()
plt.show()
