import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import contextlib
from pprint import pprint

from swarm import Swarm, PSO, SwarmGrad, SwarmGradAccel, CBO, EGICBO, PlanarSwarm

import argparse
import neptune


class SmallLinear(nn.Module):
    def __init__(self):
        super(SmallLinear, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

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

def freqs(t):
    fs = {}
    n = t.nelement()
    for i in range(10):
        fs[i] = f"{100*(t == i).sum().item()/n}%"
    return fs


def preprocess():

    # Load and preprocess the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return transform, train_dataset, test_dataset, train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Swarm Experiments on MNIST")

    parser.add_argument("--gradient", action="store_true", help="Use Adam Baseline")
    parser.add_argument("--optim", type=str, default="sga",
        choices=[
            "cbo",
            "egi",
            "pso",
            "sg",
            "sga",
            "pla",
        ],
        help="The 0th order Optim to use"
    )
    parser.add_argument("--N", type=int, default=10, help="Num Particles")

    parser.add_argument("--epo", type=int, default=1, help="Num epochs")
    parser.add_argument("--stop", type=int, default=1e15, help="Alternatively, stop after this number of batches")

    parser.add_argument("--neptune", action="store_true", help="Log to Neptune?")

    # ==== Optimizer Specific Hyperparameters ====

    parser.add_argument("--c1", type=float, default=1.0, help="c1 Hyperparam of optimizer")
    parser.add_argument("--c2", type=float, default=1.0, help="c2 Hyperparam of optimizer")
    parser.add_argument("--inertia", type=float, default=0.1, help="Inertia Hyperparam")

    # SGA
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 Hyperparam of SGA")
    parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 Hyperparam of SGA")

    # CBO
    parser.add_argument("--lamda", type=float, default=1.5, help="Lambda Hyperparam of CBO")
    parser.add_argument("--sigma", type=float, default=0.5, help="Sigma Hyperparam of CBO")
    parser.add_argument("--noise", type=str, default="component", help="Noise type of CBO")

    # EGI CBO
    parser.add_argument("--kappa", type=float, default=1e5, help="Kappa Hyperparam of EGICBO")
    parser.add_argument("--slack", type=float, default=10., help="Slack Hyperparam of EGICBO")
    parser.add_argument("--tau", type=float, default=0.2, help="Tau Hyperparam of EGICBO")
    parser.add_argument("--hess", action="store_true", help="Extrapolate using Hessian? (EGICBO)")


    return parser.parse_args()

def init_neptune(args):

    with open(".neptune_tok", "r") as f:
        tok = f.read()

    run = neptune.init_run(
        project="halcyon/swarm",
        api_token=tok,
    )

    run["parameters/gradient"] = args.gradient
    run["parameters/optim"] = args.optim
    run["parameters/N"] = args.N
    run["parameters/epochs"] = args.epo
    run["parameters/stop"] = args.stop

    return run



def main(args):

    if args.neptune:
        run = init_neptune(args)


    # Initialize the model and optimizer
    model = SmallLinear()

    if args.gradient:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
        )

    else:
        opt = args.optim
        if opt == "cbo":
            optimizer = CBO(
                model.parameters(),
                num_particles=args.N,
                lambda_=args.lamda,
                sigma=args.sigma,
                noise_type=args.noise
            )
            run["parameters/lambda"] = args.lamda
            run["parameters/sigma"] = args.sigma
            run["parameters/noise"] = args.noise

        elif opt == "pso":
            optimizer = PSO(
                model.parameters(),
                num_particles=args.N,
                c1=args.c1,
                c2=args.c2,
                inertia=args.inertia,
            )
            run["parameters/c1"] = args.c1
            run["parameters/c2"] = args.c2
            run["parameters/inertia"] = args.inertia

        elif opt == "egi":
            optimizer = EGICBO(
                model.parameters(),
                num_particles=args.N,
                lambda_=args.lamda,
                sigma=args.sigma,
                noise_type=args.noise,
                kappa=args.kappa,
                tau=args.tau,
                slack=args.slack,
                extrapolate=args.hess
            )
            run["parameters/lambda"] = args.lamda
            run["parameters/sigma"] = args.sigma
            run["parameters/kappa"] = args.kappa
            run["parameters/tau"] = args.tau
            run["parameters/slack"] = args.slack
            run["parameters/hess"] = args.hess

        elif opt == "sg":
            optimizer = SwarmGrad(
                model.parameters(),
                num_particles=args.N,
                c1=args.c1,
                c2=args.c2,
                inertia=args.inertia,
            )
            run["parameters/c1"] = args.c1
            run["parameters/c2"] = args.c2
            run["parameters/inertia"] = args.inertia

        elif opt == "sga":
            optimizer = SwarmGradAccel(
                model.parameters(),
                num_particles=args.N,
                c1=args.c1,
                c2=args.c2,
                beta1=args.beta1,
                beta2=args.beta2
            )
            run["parameters/c1"] = args.c1
            run["parameters/c2"] = args.c2
            run["parameters/inertia"] = args.inertia
            run["parameters/beta1"] = args.beta1
            run["parameters/beta2"] = args.beta2

        elif opt == "pla":
            optimizer = PlanarSwarm(
                model.parameters(),
                num_particles=args.N,
            )

        else:
            raise NotImplementedError(f"Optim={opt}")

    # Prep Data
    transform, train_dataset, test_dataset, train_loader, test_loader = preprocess()

    # Train the model

    # Dont compute gradients in case of Swarm optimizer
    train_context = torch.no_grad if not args.gradient else contextlib.nullcontext

    with train_context():
        for epoch in range(args.epo):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):

                if batch_idx+len(train_loader)*epoch > args.stop:
                    break

                if args.gradient:
                    optimizer.zero_grad()

                    output = model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()

                    optimizer.step()
                else:
                    loss = optimizer.step(lambda: F.nll_loss(model(data), target))

                    if args.neptune:
                        for stat, val in optimizer.stats().items():
                            run[f"train/{stat}"].append(val)

                if args.neptune:
                    run["train/loss"].append(loss.item())


                # if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

            # Evaluate the model after each epoch
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

                # Sanity Check: To see if net just learned to output one digit always
                pprint(freqs(pred))

            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset), accuracy))

            if args.neptune:
                run["test/loss"].append(test_loss)
                run["test/acc"].append(accuracy)

    if args.neptune:
        run.stop()


if __name__ == "__main__":

    args = parse_args()
    main(args)
