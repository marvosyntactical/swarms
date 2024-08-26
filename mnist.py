import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import contextlib
from pprint import pprint

from swarm import Swarm, PSO, SwarmGradAccel, CBO, EGICBO, PlanarSwarm
import resampling as rsmp

import argparse
import neptune


class Perceptron(nn.Module):
    # taken from https://github.com/PdIPS/CBXpy/blob/main/docs/examples/nns/models.py
    def __init__(self, mean = 0.0, std = 1.0,
                 act_fun=nn.ReLU,
                 sizes = None):
        super(Perceptron, self).__init__()

        self.mean = mean
        self.std = std
        self.act_fun = act_fun()
        self.sizes = sizes if sizes else [784, 10]
        self.linears = nn.ModuleList([nn.Linear(self.sizes[i], self.sizes[i+1]) for i in range(len(self.sizes)-1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.sizes[i+1], track_running_stats=False) for i in range(len(self.sizes)-1)])

    def __call__(self, x):
        x = x.view([x.shape[0], -1])
        x = (x - self.mean)/self.std

        for linear, bn in zip(self.linears, self.bns):
            x = linear(x)
            x = self.act_fun(x)
            x = bn(x)

        x = F.log_softmax(x, dim=1)
        return x

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

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
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
    parser.add_argument("--N", type=int, default=10, help="Number of Particles")

    parser.add_argument("--epo", type=int, default=1, help="Number of Epochs")
    parser.add_argument("--stop", type=int, default=1e15, help="Alternatively, stop after this number of batches")

    parser.add_argument("--neptune", action="store_true", help="Log to Neptune?")

    # ==== Optimizer Specific Hyperparameters ====

    parser.add_argument("--c1", type=float, default=1.0, help="c1 Hyperparameter of optimizer")
    parser.add_argument("--c2", type=float, default=1.0, help="c2 Hyperparameter of optimizer")
    parser.add_argument("--inertia", type=float, default=0.1, help="Inertia Hyperparam")
    parser.add_argument("--do-momentum", action="store_true", help="Whether to use momentum update")
    parser.add_argument("--resample", type=int, default=40, help="Resample if swarm has not improved for this many updates. Negative for no resampling.")

    # SGA
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 Hyperparameter of SGA")
    parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 Hyperparameter of SGA")
    parser.add_argument("--lr", type=float, default=1.0, help="Optional learning rate of SGA")
    parser.add_argument("--K", type=int, default=1, help="# Reference particles of SGA")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize drift")

    # CBO
    parser.add_argument("--lamda", type=float, default=1.0, help="Lambda Hyperparameter of CBO")
    parser.add_argument("--sigma", type=float, default=0.1, help="Sigma Hyperparameter of CBO")
    parser.add_argument("--noise", type=str, default="component", help="Noise type of CBO")
    parser.add_argument("--dt", type=float, default=0.1, help="dt Hyperparameter of CBO")
    parser.add_argument("--temp", type=float, default=50.0, help="Softmax Temperature")

    # EGI CBO
    parser.add_argument("--kappa", type=float, default=1e5, help="Kappa Hyperparameter of EGICBO")
    parser.add_argument("--slack", type=float, default=10., help="Slack Hyperparameter of EGICBO")
    # NOTE: TAU moved to dt
    # parser.add_argument("--tau", type=float, default=0.2, help="Tau Hyperparameter of EGICBO")
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
    else:
        run = {}


    # Initialize the model and optimizer
    model_class = Perceptron
    # model_class = SmallLinear

    if args.gradient:
        model = model_class()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
        )

    else:
        models = [model_class(sizes=[28*28,100,10]) for _ in range(args.N)]
        # models = [model_class() for _ in range(args.N)]
        model = models[0]

        if args.resample >= 0:
            rs = rsmp.resampling([rsmp.loss_update_resampling(M=1, wait_thresh=args.resample)], 1)
        else:
            rs = lambda s: None
        opt = args.optim

        if opt == "cbo":
            optimizer = CBO(
                models,
                lambda_=args.lamda,
                sigma=args.sigma,
                dt=args.dt,
                noise_type=args.noise,
                post_process=lambda cbo: rs(cbo),
                do_momentum=args.do_momentum,
                temp=args.temp
            )
            run["parameters/lambda"] = args.lamda
            run["parameters/sigma"] = args.sigma
            run["parameters/noise"] = args.noise
            run["parameters/dt"] = args.dt
            run["parameters/do_momentum"] = args.do_momentum
            run["parameters/resample"] = args.resample
            run["parameters/temp"] = args.temp

        elif opt == "pso":
            optimizer = PSO(
                models,
                c1=args.c1,
                c2=args.c2,
                inertia=args.inertia,
            )
            run["parameters/c1"] = args.c1
            run["parameters/c2"] = args.c2
            run["parameters/inertia"] = args.inertia

        elif opt == "egi":

            optimizer = EGICBO(
                models,
                lambda_=args.lamda,
                sigma=args.sigma,
                noise_type=args.noise,
                kappa=args.kappa,
                dt=args.dt,
                slack=args.slack,
                post_process=lambda cbo: rs(cbo),
                extrapolate=args.hess
            )
            run["parameters/lambda"] = args.lamda
            run["parameters/sigma"] = args.sigma
            run["parameters/kappa"] = args.kappa
            run["parameters/dt"] = args.dt
            run["parameters/slack"] = args.slack
            run["parameters/hess"] = args.hess
            run["parameters/resample"] = args.resample
            run["parameters/noise"] = args.noise

        elif opt == "sga":

            optimizer = SwarmGradAccel(
                models,
                c1=args.c1,
                c2=args.c2,
                beta1=args.beta1,
                beta2=args.beta2,
                lr=args.lr,
                K=args.K,
                do_momentum=args.do_momentum,
                post_process=lambda cbo: rs(cbo),
                normalize=args.normalize,
            )
            run["parameters/c1"] = args.c1
            run["parameters/c2"] = args.c2
            run["parameters/beta1"] = args.beta1
            run["parameters/beta2"] = args.beta2
            run["parameters/lr"] = args.lr
            run["parameters/K"] = args.K
            run["parameters/do_momentum"] = args.do_momentum
            run["parameters/normalize"] = args.normalize
            run["parameters/resample"] = args.resample

        elif opt == "pla":
            optimizer = PlanarSwarm(
                models,
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
                    loss = optimizer.step(
                        F.nll_loss,
                        model,
                        data,
                        target,
                        lambda: F.nll_loss(model(data), target)
                    )

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
