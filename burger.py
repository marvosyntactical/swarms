# Translated to PyTorch from 
# the Tensorflow implementation at https://github.com/janblechschmidt/PDEsByNNs
# See the paper at https://onlinelibrary.wiley.com/doi/pdf/10.1002/gamm.202100006

import torch
import numpy as np

from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from swarm import SwarmGradAccel, CBO
import resampling as rsmp
from scheduler import *
from torch.utils.checkpoint import checkpoint

GRADIENT = 0
DEBUG = 0
AUTOGRAD = 1
GPU = 1
OPTIM = "sga"

device = torch.device("cuda" if GPU and torch.cuda.is_available() else "cpu")

print("Using device", device)

# Set data type
DTYPE = torch.float32

# Set constants
pi = torch.tensor(np.pi, dtype=DTYPE).to(device)
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
lb = torch.tensor([tmin, xmin], dtype=DTYPE).to(device)
# Upper bounds
ub = torch.tensor([tmax, xmax], dtype=DTYPE).to(device)

# Set random seed for reproducible results
torch.manual_seed(0)

# Draw uniform sample points for initial boundary data
t_0 = torch.ones((N_0, 1), dtype=DTYPE).to(device) * lb[0]
x_0 = torch.rand((N_0, 1), dtype=DTYPE).to(device) * (ub[1] - lb[1]) + lb[1]
X_0 = torch.cat([t_0, x_0], dim=1).to(device)

# Evaluate initial condition at x_0
u_0 = fun_u_0(x_0).to(device)

# Boundary data
t_b = torch.rand((N_b, 1), dtype=DTYPE).to(device) * (ub[0] - lb[0]) + lb[0]
x_b = lb[1] + (ub[1] - lb[1]) * torch.bernoulli(torch.full((N_b, 1), 0.5, dtype=DTYPE)).to(device)
X_b = torch.cat([t_b, x_b], dim=1)

# Evaluate boundary condition at (t_b, x_b)
u_b = fun_u_b(t_b, x_b).to(device)

# Draw uniformly sampled collocation points
t_r = torch.rand((N_r, 1), dtype=DTYPE).to(device) * (ub[0] - lb[0]) + lb[0]
x_r = torch.rand((N_r, 1), dtype=DTYPE).to(device) * (ub[1] - lb[1]) + lb[1]
X_r = torch.cat([t_r, x_r], dim=1)

# Collect boundary and initial data in lists
X_data = [X_0, X_b]
u_data = [u_0, u_b]

def show_collocations():
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(t_0.cpu().numpy(), x_0.cpu().numpy(), c=u_0.cpu().numpy(), marker='X', vmin=-1, vmax=1)
    plt.scatter(t_b.cpu().numpy(), x_b.cpu().numpy(), c=u_b.cpu().numpy(), marker='X', vmin=-1, vmax=1)
    plt.scatter(t_r.cpu().numpy(), x_r.cpu().numpy(), c='r', marker='.', alpha=0.1)
    plt.xlabel('$t$')
    plt.ylabel('$x$')

    plt.title('Positions of collocation points and boundary data')
    # plt.savefig('Xdata_Burgers.pdf', bbox_inches='tight', dpi=300)
    # plt.show()

class ScalingLayer(torch.nn.Module):
    def __init__(self, lb, ub, device):
        super(ScalingLayer, self).__init__()
        self.lb = lb.view(1, -1).to(device)
        self.ub = ub.view(1, -1).to(device)

    def forward(self, x):
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0


def init_model(num_hidden_layers=8, num_neurons_per_layer=20):
    # Initialize a feedforward neural network
    layers = []

    def make_dbg(s):
        def dbg(layer, x):
            if DEBUG:
                print(s)
                if isinstance(layer, torch.nn.Linear):
                    assert layer.weight.requires_grad, s
                    assert layer.bias.requires_grad, s
                    print("w", layer.weight.device)
                    print("b", layer.bias.device)
                assert x[0].requires_grad, s
                print("x", x[0].device)
        return dbg

    # input_layer = torch.nn.Linear(2, num_neurons_per_layer)
    # layers.append(input_layer)

    # Introduce a scaling layer to map input to [lb, ub]
    scaling_layer = ScalingLayer(lb, ub, device)
    scaling_layer.register_forward_pre_hook(make_dbg("scaling"))
    layers.append(scaling_layer)

    input_layer = torch.nn.Linear(2, num_neurons_per_layer)
    # input_layer.weight.data = input_layer.weight.data.to(device)
    input_layer.register_forward_pre_hook(make_dbg("input"))
    layers.append(input_layer)

    # Append hidden layers
    for i in range(num_hidden_layers):
        hidden_layer = torch.nn.Linear(
            num_neurons_per_layer, num_neurons_per_layer
        )
        hidden_layer = hidden_layer
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

    assert u.requires_grad, u.requires_grad

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
    model = init_model().to(device)

    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        milestones=[1000, 3000],
        gamma=0.1
    )
    for n, p in model.named_parameters():
        print(f"{n}: {p.requires_grad}")
else:
    N = 50
    K = 5
    models = [init_model().to(device) for _ in range(N)]

    for m in models:
        for n, p in m.named_parameters():
            p.requires_grad_(True)
            # print(p.device)

    if OPTIM == "sga":
        optimizer = SwarmGradAccel(
            models,
            c1=1,
            c2=0,
            lr=0.1,
            beta1=0.7,
            beta2=0.9,
            K=K,
            do_momentum=True,
            normalize=2,
            # post_process=lambda o: rs(o),
            parallel=not AUTOGRAD,
            device=device
        )
    elif OPTIM == "cbo":

        rs = rsmp.resampling([rsmp.loss_update_resampling(M=1, wait_thresh=40, device=device)], 1)

        optimizer = CBO(
            models,
            dt=.1,
            post_process=lambda cbo: rs(cbo),
            do_momentum=True,
            parallel=not AUTOGRAD,
            device=device,
        )

    model = optimizer.model

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

            # print("t requires grad:", t.requires_grad)
            # print("x requires grad:", x.requires_grad)
            # print("stackered requires grad:", stackered.requires_grad)

            # print("u requires grad:", u.requires_grad)
            # print("u_pred_0 requires grad:", u_pred_0.requires_grad)
            # print("u_pred_b requires grad:", u_pred_b.requires_grad)

            def compute_u_t(u, t):
                return torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            def compute_u_x(u, x):
                return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            def compute_u_xx(u_x, x):
                return torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x))[0]

            # Compute gradients
            # print(f"Before autograd")
            u_t = compute_u_t(u, t)
            # print(f"After autograd 0")
            u_x = compute_u_x(u, x)
            # print(f"After autograd 1")
            u_xx = compute_u_xx(u_x, x) # do not need graph further, so dont need to ckpt
            # print(f"After autograd 2")

            residual = fun_r(t, x, u, u_t, u_x, u_xx)

            u_t.detach()
            u_x.detach()
            u_xx.detach()

            del u, u_t, u_x, u_xx

            # Initialize loss
            loss = torch.mean(torch.square(residual))

            # Add phi^0 and phi^b to the loss
            loss += torch.mean(torch.square(u_data[0] - u_pred_0))
            loss += torch.mean(torch.square(u_data[1] - u_pred_b))

            return loss

        with torch.enable_grad():
            loss = optimizer.step(
                model,
                stackered,
                get_loss_
            )

        scheduler.step()
        optimizer.zero_grad() # when should I call zero grad?

        with torch.no_grad():
            for param in optimizer.model.parameters():
                param.detach_()

    return loss

# Number of training epochs
epochs = 300
hist = []

# Start timer
t0 = time()

for i in range(epochs):

    loss = train_step()

    # Append current loss to hist
    hist.append(loss.item())

    # if i%50 == 0:
    print('It {:05d}: loss = {:10.8e}'.format(i, loss.item()))

    del loss

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
U = upred.cpu().numpy().reshape(n_eval+1, n_eval+1)

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
