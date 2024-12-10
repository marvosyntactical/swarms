import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, 512)
        self.output = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Define the ODE and boundary conditions
# def ode(x, y):
#     return y.diff(x) - (x**2 + 1) * y
def ode(x, y):
    # Compute dy/dx using autograd
    dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return dy_dx - (x**2 + 1) * y

def boundary_conditions(x0, y0, x1, y1):
    return y0 - 1, y1 - torch.exp(0.5 * (x1**2 + 2 * x1))

# Generate training data
x = torch.linspace(0, 1, 100).reshape(-1, 1)
x.requires_grad_(True)

# Initialize the network and optimizer
net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)

lamda = 1

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Sample random points from the domain
    x_random = torch.rand_like(x) * (x[-1] - x[0]) + x[0]
    x_random.requires_grad_(True)

    # Forward pass
    y_random = net(x_random)
    y = net(x)

    # Compute the loss
    ode_loss = torch.mean(ode(x_random, y_random)**2)
    bc_loss = torch.sum(torch.tensor(boundary_conditions(x[0], y[0], x[-1], y[-1]))**2)
    loss = ode_loss + lamda * bc_loss

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Test the trained network
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
y_test = net(x_test).detach().numpy()

# Plot the results
plt.plot(x_test.numpy(), y_test, label='Predicted')
plt.plot(x_test.numpy(), np.exp(0.5 * (x_test.numpy()**2 + 2 * x_test.numpy())), label='Exact')
plt.legend()
plt.show()
