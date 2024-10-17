import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import contextlib
from pprint import pprint

from swarm import Swarm, PSO, SwarmGradAccel, CBO, EGICBO, PlanarSwarm
from scheduler import *
import resampling as rsmp

import argparse
import neptune

class ObjectiveFunction(nn.Module):
    def __init__(self, func, dim, mean, std):
        super(ObjectiveFunction, self).__init__()
        assert mean.shape[0] == dim, (mean.shape, dim)
        self.func = func
        self.param = nn.Parameter(torch.randn(dim)*std+mean)

    def __call__(self, x):
        return self.func(self.param)


def rastrigin(x):
    return 10 * len(x) + torch.sum(x**2 - 10 * torch.cos(2 * torch.pi * x))


def ackley(x):
    n = len(x)
    return -20 * torch.exp(-0.2 * torch.sqrt(torch.sum(x**2) / n)) - \
           torch.exp(torch.sum(torch.cos(2 * torch.pi * x)) / n) + 20 + torch.e


def xsy4(x):
    sines = torch.sum(torch.sin(x)**2)
    sinesqrts = torch.sum(x**2)
    squares = torch.sum(torch.sin(torch.sqrt(torch.sqrt(x**2)))**2)
    return (sines - torch.exp(-squares)) * torch.exp(-sinesqrts)


def griewank(x):
    summands = torch.linalg.norm(x)**2
    div = (torch.arange(len(x))+1)**.5
    factors = torch.prod(torch.cos(x/div))
    return 1 + summands/4000 + factors


def sphere(x):
    return torch.linalg.norm(x)**2


def parse_args():
    parser = argparse.ArgumentParser(description="Swarm Experiments on MNIST")

    parser.add_argument("--gradient", action="store_true", help="Use Adam Baseline.")
    parser.add_argument("--optim", type=str, default="sga",
        choices=[
            "cbo",
            "egi",
            "pso",
            "sga",
            "pla",
        ],
        help="The 0th order Optim to use."
    )
    parser.add_argument("--objective", type=str, default="ackley",
        choices=[
            "ackley",
            "rastrigin",
            "xsy4",
            "sphere",
            "griewank",
        ],
        help="The function to optimize."
    )
    parser.add_argument("--N", type=int, default=50, help="Number of Particles")
    parser.add_argument("--dim", type=int, default=100, help="Number of Dimensions")
    parser.add_argument("--iterations", type=float, default=5e3, help="Number of Steps to do.")
    parser.add_argument("--std", type=float, default=5., help="Standard Deviation of Init Normal")
    parser.add_argument("--means", nargs="+", type=float, default=[85.], help="Init location for subswarm i is [means[i], ..., mean[i]].")

    parser.add_argument("--neptune", action="store_true", help="Log to Neptune?")

    # ==== Optimizer Specific Hyperparameters ====

    parser.add_argument("--c1", type=float, default=1.0, help="c1 Hyperparameter of optimizer")
    parser.add_argument("--c2", type=float, default=1.0, help="c2 Hyperparameter of optimizer")
    parser.add_argument("--inertia", type=float, default=0.1, help="Inertia Hyperparam")
    parser.add_argument("--resample", type=int, default=40, help="Resample if swarm has not improved for this many updates. Negative for no resampling.")

    # Momentum
    parser.add_argument("--do-momentum", action="store_true", help="Whether to use momentum update")
    parser.add_argument("--beta1", type=float, default=0.7, help="Beta1 Hyperparameter of SGA")
    parser.add_argument("--beta2", type=float, default=0.9, help="Beta2 Hyperparameter of SGA")

    # SGA
    parser.add_argument("--lr", type=float, default=1.0, help="Optional learning rate of SGA")
    parser.add_argument("--K", type=int, default=1, help="# Reference particles of SGA")
    parser.add_argument("--normalize", type=int, default=0, help="Whether to normalize drift by h**norm")
    parser.add_argument("--sub", type=int, default=1, help="Number of sub swarms (flat hierachy). Must divide --N evenly.")
    parser.add_argument("--leak", type=float, default=0.1, help="Leak slope of Leaky Relu in SGA")

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


    return parser.parse_args()


def init_neptune(args):

    with open(".neptune_tok", "r") as f:
        tok = f.read()

    run = neptune.init_run(
        project="halcyon/benchswarm",
        api_token=tok,
    )

    run["parameters/objective"] = args.objective
    run["parameters/gradient"] = args.gradient
    run["parameters/optim"] = args.optim
    run["parameters/N"] = args.N
    run["parameters/dim"] = args.dim

    return run



def main(args):

    if args.neptune:
        run = init_neptune(args)
    else:
        run = {}

    if args.gradient:
        # TODO 
        raise NotImplementedError("Implement optimization using ADAM")

    else:
        objective = eval(args.objective) # NOTE super unsafe, switch to dict

        assert len(args.means) == args.sub, (args.means, args.sub)
        offsets = [torch.Tensor([args.means[i]]*args.dim) for i in range(args.sub)]
        subswarm = lambda prtcl: int(prtcl//(args.N/args.sub))
        models = [ObjectiveFunction(objective, args.dim, offsets[subswarm(prtcl)], args.std) for prtcl in range(args.N)]
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
                leak=args.leak,
                do_momentum=args.do_momentum,
                post_process=lambda cbo: rs(cbo),
                normalize=args.normalize,
                sub_swarms=args.sub
            )
            run["parameters/c1"] = args.c1
            run["parameters/c2"] = args.c2
            run["parameters/beta1"] = args.beta1
            run["parameters/beta2"] = args.beta2
            run["parameters/lr"] = args.lr
            run["parameters/leak"] = args.leak
            run["parameters/K"] = args.K
            run["parameters/do_momentum"] = args.do_momentum
            run["parameters/normalize"] = args.normalize
            run["parameters/resample"] = args.resample
            run["parameters/sub_swarms"] = args.sub

        elif opt == "pla":
            optimizer = PlanarSwarm(
                models,
            )
        else:
            raise NotImplementedError(f"Optim={opt}")

        # SCHEDULER
        run["parameters/scheduler"] = args.sched
        run["parameters/sched_hyper"] = args.sched_hyper

        if args.sched == "none":
            scheduler = NoScheduler(optimizer)
        elif args.sched == "cos":
            scheduler = CosineAnnealingLR(optimizer, args.sched_hyper) # T_max
        elif args.sched == "exp":
            scheduler = ExponentialLR(optimizer, args.sched_hyper) # gamma
        elif args.sched == "step":
            scheduler = StepLR(optimizer, args.sched_hyper) # step_size
        elif args.sched == "plat":
            scheduler = ReduceLROnPlateau(optimizer, patience=args.sched_hyper) # patience
        else:
            raise NotImplementedError(f"Scheduler {args.sched} not implemented.")

    # Dont compute gradients in case of Swarm optimizer
    train_context = torch.no_grad if not args.gradient else contextlib.nullcontext

    with train_context():
        model.train()
        for it in range(1, int(args.iterations)+1):

            if args.gradient:
                raise NotImplementedError(f"Training loop for ADAM")
                optimizer.zero_grad()

                # output = model(data)
                # loss = F.nll_loss(output, target)
                # loss.backward()

                optimizer.step()
            else:
                dummy = torch.Tensor([42])
                loss = optimizer.step(
                    model,
                    dummy,
                    lambda x: x,
                )
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(loss)
                else:
                    scheduler.step()

                if args.neptune:
                    for stat, val in optimizer.stats().items():
                        run[f"train/{stat}"].append(val)

            if args.neptune:
                run["train/loss"].append(loss.item())

            norm = torch.linalg.norm(model.param).item()
            print(f"Iter:\t{it}/{int(args.iterations)}\tLoss:\t({loss.item():.4f})\tNorm:\t{norm:.4f}")

    if args.neptune:
        run.stop()


if __name__ == "__main__":

    args = parse_args()
    main(args)
