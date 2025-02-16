import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import contextlib
from pprint import pprint

from copy import deepcopy

from swarm import Swarm, PSO, SwarmGradAccel, CBO, EGICBO, PlanarSwarm, DiffusionEvolution, GradSwarm
from scheduler import *
import resampling as rsmp

from diffevo.fitnessmapping import *

import argparse
import neptune


def time(optimizer, model, data_loader, n, device):
    import timeit

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        break

    t = timeit.timeit(lambda: optimizer.step(model, data, lambda out: F.nll_loss(out,target)), number=n) / n

    print(f"Average step time for Optimizer: {t:.6f} seconds")


class SmallConvNet(nn.Module):
    def __init__(self, sizes=None):
        super(SmallConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(7 * 7 * 16, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 7 * 7 * 16)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class Perceptron(nn.Module):
    # taken from https://github.com/PdIPS/CBXpy/blob/main/docs/examples/nns/models.py
    def __init__(
            self, mean = 0.0, std = 1.0,
            act_fun=nn.ReLU,
            sizes = None
        ):
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

    parser.add_argument("--gradient", action="store_true", help="Use Adam Baseline.")
    parser.add_argument("--arch", type=str, default="mlp",
        choices=[
            "mlp",
            "conv",
        ]
    )
    parser.add_argument("--optim", type=str, default="sga",
        choices=[
            "cbo",
            "egi",
            "pso",
            "sga",
            "pla",
            "evo",
            "gsa"
        ],
        help="The 0th order Optim to use."
    )
    parser.add_argument("--N", type=int, default=50, help="Number of Particles")

    parser.add_argument("--epo", type=int, default=10, help="Number of Epochs")
    parser.add_argument("--switch", type=int, default=-1, help="Switch from 0 Order optimizer to Adam after this many batches")
    parser.add_argument("--stop", type=int, default=1e15, help="End Epochs after this number of batches")
    parser.add_argument("--hidden", type=int, default=100, help="Width of hidden layer of MLP with \
    architecture [784,args.hidden,10]")

    parser.add_argument("--neptune", action="store_true", help="Log to Neptune?")
    parser.add_argument("--gpu", action="store_true", help="Try to use CUDA?")
    parser.add_argument("--resample", type=int, default=40, help="Resample if swarm has not improved for this many updates. Negative for no resampling.")

    # ==== Optimizer Specific Hyperparameters ====

    # PSO
    parser.add_argument("--c1", type=float, default=1.0, help="c1 Hyperparameter of optimizer")
    parser.add_argument("--c2", type=float, default=1.0, help="c2 Hyperparameter of optimizer")
    parser.add_argument("--inertia", type=float, default=0.1, help="Inertia Hyperparam")

    # Momentum
    parser.add_argument("--do-momentum", action="store_true", help="Whether to use momentum update")
    parser.add_argument("--beta1", type=float, default=0.7, help="Beta1 Hyperparameter of SGA")
    parser.add_argument("--beta2", type=float, default=0.9, help="Beta2 Hyperparameter of SGA")

    # SGA
    parser.add_argument("--leak", type=float, default=0.1, help="Leak of SGA")
    parser.add_argument("--lr", type=float, default=1.0, help="Optional learning rate of SGA")
    parser.add_argument("--K", type=int, default=1, help="# Reference particles of SGA")
    parser.add_argument("--normalize", type=int, default=2, help="Whether to normalize drift by h**normalize")
    parser.add_argument("--sub", type=int, default=1, help="Number of sub swarms (flat hierachy). Must divide --N evenly.")

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
    parser.add_argument("--hess", action="store_true", help="Extrapolate using Hessian (currently intractable)? (EGICBO)")

    # DIFF EVO
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps (DiffEvo)")
    parser.add_argument("--temperature", type=float, default=0.0, help="If != 0, use Energy Mapping \
            with this temp (DiffEvo)")
    parser.add_argument("--ddim-noise", type=float, default=1.0, help="Noise of generator sample \
            (<=1) (DiffEvo)")
    parser.add_argument("--l2", type=float, default=0.0, help="l2 penalty (DiffEvo)")
    parser.add_argument("--latent", type=int, default=0, help="Latent dim of Latent DiffEvo (ignored if 0)")

    # GRAD SWARM
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup steps for GradSwarm")

    parser.add_argument("--timeit", type=int, default=-1, help="If > 0, avg execution time of optimizer.step() over this many executions")

    # Scheduler
    parser.add_argument("--sched", type=str, default="none", help="Scheduler Class used.",
        choices=[
            "none",
            "step",
            "exp",
            "cos",
            "plat",
        ]
    )
    parser.add_argument("--sched-hyper", type=float, default=.1, help="Scheduler Hyperparameter")
    parser.add_argument("--autograd", action="store_true", help="Do NOT parallelize models using vmap? enables autograd")

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
    run["parameters/gpu"] = args.gpu

    return run



def main(args):

    if args.neptune:
        run = init_neptune(args)
    else:
        run = {}


    # Initialize the model and optimizer
    if args.arch == "mlp":
        model_class = Perceptron
    elif args.arch == "conv":
        model_class = SmallConvNet
    else:
        raise NotImplementedError(f"Architecture {args.arch}")

    # model_class = SmallLinear

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # sizes = [28*28, args.hidden, 10]
    sizes = [28*28, 10]

    if not args.gradient:

        models = [model_class(sizes=sizes).to(device) for _ in range(args.N)]
        # models = [model_class() for _ in range(args.N)]

        if args.resample >= 0:
            rs = rsmp.resampling([rsmp.loss_update_resampling(M=1, wait_thresh=args.resample,
                device=device)], 1)
        else:
            rs = lambda s: None
        opt = args.optim

        if opt == "cbo":
            optimizer0 = CBO(
                models,
                lambda_=args.lamda,
                sigma=args.sigma,
                dt=args.dt,
                noise_type=args.noise,
                post_process=lambda cbo: rs(cbo),
                do_momentum=args.do_momentum,
                temp=args.temp,
                device=device,
                parallel=not args.autograd
            )
            run["parameters/lambda"] = args.lamda
            run["parameters/sigma"] = args.sigma
            run["parameters/noise"] = args.noise
            run["parameters/dt"] = args.dt
            run["parameters/do_momentum"] = args.do_momentum
            run["parameters/resample"] = args.resample
            run["parameters/temp"] = args.temp

        elif opt == "pso":
            optimizer0 = PSO(
                models,
                device=device,
                c1=args.c1,
                c2=args.c2,
                inertia=args.inertia,
            )
            run["parameters/c1"] = args.c1
            run["parameters/c2"] = args.c2
            run["parameters/inertia"] = args.inertia

        elif opt == "egi":

            optimizer0 = EGICBO(
                models,
                device=device,
                lambda_=args.lamda,
                sigma=args.sigma,
                noise_type=args.noise,
                kappa=args.kappa,
                dt=args.dt,
                slack=args.slack,
                post_process=lambda cbo: rs(cbo),
                extrapolate=args.hess,
                parallel=not args.autograd
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

            optimizer0 = SwarmGradAccel(
                models,
                device=device,
                c1=args.c1,
                c2=args.c2,
                beta1=args.beta1,
                beta2=args.beta2,
                leak=args.leak,
                lr=args.lr,
                K=args.K,
                do_momentum=args.do_momentum,
                post_process=lambda cbo: rs(cbo),
                normalize=args.normalize,
                sub_swarms=args.sub,
                parallel=not args.autograd
            )
            run["parameters/c1"] = args.c1
            run["parameters/c2"] = args.c2
            run["parameters/beta1"] = args.beta1
            run["parameters/beta2"] = args.beta2
            run["parameters/leak"] = args.leak
            run["parameters/lr"] = args.lr
            run["parameters/K"] = args.K
            run["parameters/do_momentum"] = args.do_momentum
            run["parameters/normalize"] = args.normalize
            run["parameters/resample"] = args.resample
            run["parameters/sub_swarms"] = args.sub

        elif opt == "pla":
            optimizer0 = PlanarSwarm(
                models,
                device=device,
            )
        elif opt == "evo":
            if args.temperature:
                fitness_mapping = Energy(temperature=args.temperature, l2_factor=args.l2)
            else:
                fitness_mapping = Identity(l2_factor=args.l2)

            optimizer0 = DiffusionEvolution(
                models,
                device=device,
                num_steps=args.num_steps,
                noise=args.ddim_noise,
                fitness_mapping=fitness_mapping,
                latent_dim=args.latent if args.latent else None,
                parallel=not args.autograd
            )
        elif opt == "gsa":
            optimizer0 = GradSwarm(
                models,
                device=device,
                opt_args={"lr": args.lr},
                warmup=args.warmup
            )
        else:
            raise NotImplementedError(f"Optim={opt}")

        model = optimizer0.model

        # SCHEDULER
        run["parameters/scheduler"] = args.sched
        run["parameters/sched_hyper"] = args.sched_hyper

        if args.sched == "none":
            scheduler = NoScheduler(optimizer0)
        elif args.sched == "cos":
            scheduler = CosineAnnealingLR(optimizer0, args.sched_hyper) # T_max
        elif args.sched == "exp":
            scheduler = ExponentialLR(optimizer0, args.sched_hyper) # gamma
        elif args.sched == "step":
            scheduler = StepLR(optimizer0, args.sched_hyper) # step_size
        elif args.sched == "plat":
            scheduler = ReduceLROnPlateau(optimizer0, patience=args.sched_hyper) # patience
        else:
            raise NotImplementedError(f"Scheduler {args.sched} not implemented.")

    if args.gradient:
        model = model_class(sizes=sizes).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )

    # Prep Data
    transform, train_dataset, test_dataset, train_loader, test_loader = preprocess()

    # Train the model

    # Dont compute gradients in case of Swarm optimizer
    train_context = torch.no_grad if not (args.gradient or (args.switch >=0) or args.optim == "gsa") else contextlib.nullcontext

    if args.timeit > 0:
        time(optimizer, model, train_loader, args.timeit, device)

    gradient = args.gradient

    with train_context():
        for epoch in range(args.epo):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                if batch_idx + epoch*len(train_loader) == args.switch:
                    gradient = True
                    model = deepcopy(model)
                    model.train()
                    for p in model.parameters():
                        p.requires_grad_(True)

                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=0.01,
                    )

                if batch_idx + epoch*len(train_loader) > args.stop:
                    break

                if gradient:
                    optimizer.zero_grad()

                    output = model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()

                    optimizer.step()
                else:
                    loss = optimizer0.step(
                        model,
                        data,
                        lambda out: F.nll_loss(out, target)
                    )

                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(loss)
                    else:
                        scheduler.step()

                    if args.neptune:
                        for stat, val in optimizer0.stats().items():
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
                data, target = data.to(device), target.to(device)
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
