import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import contextlib

from swarm import Swarm, PSO, SwarmGrad, SwarmGradAccel


GRADIENT = 0 # use zeroth order optim (swarm) or 1st order optim (adam) ?

if GRADIENT:
    Optim = torch.optim.Adam
else:
    # Optim = Swarm
    # Optim = PSO
    # Optim = SwarmGrad
    Optim = SwarmGradAccel

    N = 10

# Define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def main(args):
    # Load and preprocess the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize the model and optimizer
    model = Net()
    if GRADIENT:
        optimizer = Optim(
            model.parameters(),
            lr=0.01,
        )
    else:
        optimizer = Optim(
            model.parameters(),
            num_particles=N
        )


    # Train the model
    num_epochs = 10

    # Dont compute gradients in case of Swarm optimizer
    train_context = torch.no_grad if not GRADIENT else contextlib.nullcontext

    with train_context():
        for epoch in range(num_epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):

                if GRADIENT:
                    optimizer.zero_grad()

                    output = model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()

                    optimizer.step()

                else:
                    loss = optimizer.step(lambda: F.nll_loss(model(data), target))

                # if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    # Evaluate the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
