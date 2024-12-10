# Translated to PyTorch from 
# the Tensorflow implementation at https://github.com/janblechschmidt/PDEsByNNs
# See the paper at https://onlinelibrary.wiley.com/doi/pdf/10.1002/gamm.202100006

import torch
import numpy as np

from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from swarm import SwarmGradAccel
import resampling as rsmp
from scheduler import *

GRADIENT = 0


# Set data type
DTYPE = torch.float32

# Set constants
pi = torch.tensor(np.pi, dtype=DTYPE)
viscosity = .01 / pi

# Define initial condition
def fun_u_0(x):
    return -torch.sin(pi * x)

# Define boundary condition
def fun_u_b(t, x):
    n = x.shape[0]
    return torch.zeros((n, 1), dtype=DTYPE)

# Define residual of the PDE
def fun_r(t, x, u, u_t, u_x, u_xx):
    return u_t + u * u_x - viscosity * u_xx

# Set number of data points
N_0 = 50
N_b = 50
N_r = 10000

# Set boundary
tmin = 0.
tmax = 1.
xmin = -1.
xmax = 1.

# Lower bounds
lb = torch.tensor([tmin, xmin], dtype=DTYPE)
# Upper bounds
ub = torch.tensor([tmax, xmax], dtype=DTYPE)

# Set random seed for reproducible results
torch.manual_seed(0)

# Draw uniform sample points for initial boundary data
t_0 = torch.ones((N_0, 1), dtype=DTYPE) * lb[0]
x_0 = torch.rand((N_0, 1), dtype=DTYPE) * (ub[1] - lb[1]) + lb[1]
X_0 = torch.cat([t_0, x_0], dim=1)

# Evaluate initial condition at x_0
u_0 = fun_u_0(x_0)

# Boundary data
t_b = torch.rand((N_b, 1), dtype=DTYPE) * (ub[0] - lb[0]) + lb[0]
x_b = lb[1] + (ub[1] - lb[1]) * torch.bernoulli(torch.full((N_b, 1), 0.5, dtype=DTYPE))
X_b = torch.cat([t_b, x_b], dim=1)

# Evaluate boundary condition at (t_b, x_b)
u_b = fun_u_b(t_b, x_b)

# Draw uniformly sampled collocation points
t_r = torch.rand((N_r, 1), dtype=DTYPE) * (ub[0] - lb[0]) + lb[0]
x_r = torch.rand((N_r, 1), dtype=DTYPE) * (ub[1] - lb[1]) + lb[1]
X_r = torch.cat([t_r, x_r], dim=1)

# Collect boundary and initial data in lists
X_data = [X_0, X_b]
u_data = [u_0, u_b]

def show_collocations():
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(t_0.numpy(), x_0.numpy(), c=u_0.numpy(), marker='X', vmin=-1, vmax=1)
    plt.scatter(t_b.numpy(), x_b.numpy(), c=u_b.numpy(), marker='X', vmin=-1, vmax=1)
    plt.scatter(t_r.numpy(), x_r.numpy(), c='r', marker='.', alpha=0.1)
    plt.xlabel('$t$')
    plt.ylabel('$x$')

    plt.title('Positions of collocation points and boundary data')
    # plt.savefig('Xdata_Burgers.pdf', bbox_inches='tight', dpi=300)
    # plt.show()

class ScalingLayer(torch.nn.Module):
    def __init__(self, lb, ub):
        super(ScalingLayer, self).__init__()
        self.lb = lb.view(1, -1)
        self.ub = ub.view(1, -1)

    def forward(self, x):
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0


def init_model(num_hidden_layers=8, num_neurons_per_layer=20):
    # Initialize a feedforward neural network
    layers = []

    def make_dbg(s):
        def dbg(layer, x):
            assert x[0].requires_grad, s
        return dbg

    # input_layer = torch.nn.Linear(2, num_neurons_per_layer)
    # layers.append(input_layer)

    # Introduce a scaling layer to map input to [lb, ub]
    scaling_layer = ScalingLayer(lb, ub)
    scaling_layer.register_forward_pre_hook(make_dbg("scaling"))
    layers.append(scaling_layer)

    input_layer = torch.nn.Linear(2, num_neurons_per_layer)
    input_layer.register_forward_pre_hook(make_dbg("input"))
    layers.append(input_layer)


    # Append hidden layers
    for i in range(num_hidden_layers):
        hidden_layer = torch.nn.Linear(
            num_neurons_per_layer, num_neurons_per_layer
        )
        hidden_layer.register_forward_pre_hook(make_dbg(i))
        layers.append(hidden_layer)
        layers.append(torch.nn.Tanh())

    # Output is one-dimensional
    output_layer = torch.nn.Linear(num_neurons_per_layer, 1)
    layers.append(output_layer)

    return torch.nn.Sequential(*layers)

def get_r(model, X_r):
    # Enable gradient computation
    X_r.requires_grad_(True)

    # Split t and x to compute partial derivatives
    t, x = X_r[:, 0:1], X_r[:, 1:2]

    stacked = torch.stack([t[:, 0], x[:, 0]], dim=1)

    # Determine residual
    u = model(stacked)

    # Compute gradients
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    return fun_r(t, x, u, u_t, u_x, u_xx)

def compute_loss(model, X_r, X_data, u_data):
    # Compute phi^r
    r = get_r(model, X_r)
    phi_r = torch.mean(torch.square(r))

    # Initialize loss
    loss = phi_r

    # Add phi^0 and phi^b to the loss
    for i in range(len(X_data)):
        u_pred = model(X_data[i])
        loss += torch.mean(torch.square(u_data[i] - u_pred))

    return loss



if GRADIENT:
    # Initialize model aka u_\theta
    model = init_model()

    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        milestones=[1000, 3000],
        gamma=0.1
    )
else:
    N = 50
    K = 5
    models = [init_model() for _ in range(N)]
    model = models[0]
    for m in models:
        for p in model.parameters():
            p.requires_grad_(True)

    rs = rsmp.resampling([rsmp.loss_update_resampling(M=1, wait_thresh=40)], 1)

    optimizer = SwarmGradAccel(
        models,
        c1=1,
        c2=0,
        beta1=0.7,
        beta2=0.9,
        K=K,
        do_momentum=True,
        normalize=2,
        post_process=lambda o: rs(o),
    )
    scheduler = StepLR(optimizer, 100, gamma=0.1) # step_size




def train_step():

    if GRADIENT:
        # Compute current loss and gradient w.r.t. parameters
        loss = compute_loss(model, X_r, X_data, u_data)

        # Compute gradients
        loss.backward()

        # Perform gradient descent step
        lr_schedule.optimizer.step()

        # Zero the gradients to avoid accumulation
        lr_schedule.optimizer.zero_grad()

        # Update learning rate schedule
        lr_schedule.step()
    else:

        X_r.requires_grad_(True)
        X_data[0].requires_grad_(True)
        X_data[1].requires_grad_(True)

        t, x = X_r[:, 0:1], X_r[:, 1:2]
        # t.requires_grad_(True)
        # x.requires_grad_(True)

        stacked = torch.stack([t[:, 0], x[:, 0]], dim=1)
        stackered = torch.vstack([stacked, X_data[0], X_data[1]])
        # stackered.requires_grad_(True)

        def get_loss_(model_preds):
            u = model_preds[:N_r]
            u_pred_0 = model_preds[N_r:N_r+N_0]
            u_pred_b = model_preds[N_r+N_0:]

            print("u requires grad:", u.requires_grad)
            print("u_pred_0 requires grad:", u_pred_0.requires_grad)
            print("u_pred_b requires grad:", u_pred_b.requires_grad)

            # Compute gradients
            u_t = torch.autograd.grad(
                u, t, grad_outputs=torch.ones_like(u), create_graph=True
            )[0]
            u_x = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0]
            u_xx = torch.autograd.grad(
                u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
            )[0]

            residual = fun_r(t, x, u, u_t, u_x, u_xx)

            # Initialize loss
            loss = torch.mean(torch.square(residual))

            # Add phi^0 and phi^b to the loss
            loss += torch.mean(torch.square(u_data[0] - u_pred_0))
            loss += torch.mean(torch.square(u_data[i] - u_pred_b))

            return loss

        with torch.enable_grad():
            loss = optimizer.step(
                model,
                stackered,
                get_loss_
            )


        scheduler.step()


    return loss

# Number of training epochs
epochs = 300
hist = []

# Start timer
t0 = time()

for i in range(epochs+1):

    loss = train_step()

    # Append current loss to hist
    hist.append(loss.item())


    # Output current loss after 50 iterates
    if i%50 == 0:
        print('It {:05d}: loss = {:10.8e}'.format(i, loss.item()))

# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))


# Set up meshgrid
n_eval = 600
tspace = np.linspace(lb[0], ub[0], n_eval + 1)
xspace = np.linspace(lb[1], ub[1], n_eval + 1)
T, X = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T.flatten(), X.flatten()]).T

# Convert Xgrid to PyTorch tensor
Xgrid_tensor = torch.tensor(Xgrid, dtype=DTYPE)

# Determine predictions of u(t, x)
with torch.no_grad():
    upred = model(Xgrid_tensor)

# Reshape upred
U = upred.numpy().reshape(n_eval+1, n_eval+1)

# Surface plot of solution u(t,x)
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, X, U, cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_zlabel('$u_\\theta(t,x)$')
ax.set_title('Solution of Burgers equation')
# plt.savefig('Burgers_Solution.pdf', bbox_inches='tight', dpi=300)
plt.show()

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
ax.semilogy(range(len(hist)), hist, 'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi_{n_{epoch}}$')
plt.show()
