import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
trainset = datasets.MNIST('data/', download=True, train=True, transform=transform)
testset = datasets.MNIST('data/', download=True, train=False, transform=transform)

# create the dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# define the network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128) # 28*28=784
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10) # 10 classes

    def forward(self, x):
        # flatten the image
        x = x.view(x.shape[0], -1)
        # add hidden layers with relu activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # add output layer with softmax activation
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
    
model = Net()
print(model)


# define the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


# define the training loop, collect the loss and accuracy for each epoch
epochs = 10
train_losses, test_losses, test_accuracy = [], [], []


for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # zero the gradients
        optimizer.zero_grad()
        # forward pass
        log_ps = model(images)
        # calculate loss
        loss = criterion(log_ps, labels)
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()
        # update running loss
        running_loss += loss.item()
    else:
        # calculate test loss and accuracy
        test_loss = 0
        accuracy = 0
        # turn off gradients for validation
        with torch.no_grad():
            for images, labels in testloader:
                # forward pass
                log_ps = model(images)
                # calculate loss
                test_loss += criterion(log_ps, labels)
                # calculate accuracy
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        # store the losses and accuracy for each epoch
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        test_accuracy.append(accuracy/len(testloader))
        # print the losses and accuracy for each epoch
        print("Epoch: {}/{} ".format(e+1, epochs),
              "Training Loss: {:.4f} ".format(train_losses[-1]),
              "Test Loss: {:.4f} ".format(test_losses[-1]),
              "Test Accuracy: {:.4f}".format(test_accuracy[-1]))
        

# plot the losses
fig = plt.figure(figsize=(6, 4))
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

# plot the accuracy
fig = plt.figure(figsize=(6, 4))
plt.plot(test_accuracy, label='Accuracy')
plt.legend(frameon=False)
plt.show()
