import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from swarm import Swarm, PSO, SwarmGradAccel, CBO, EGICBO, PlanarSwarm
from scheduler import *
import resampling as rsmp

import argparse
import neptune
import contextlib
import matplotlib.pyplot as plt


# Define the small neural network
class RegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionNet, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.output(x)
        return x

# Generate random data for regression
input_dim = 10
output_dim = 10
num_samples = 1000

# functional relationship:
f = lambda x: 2 * x + 1

X = torch.rand(num_samples, input_dim) * 10 - 5
Y = f(X) # + torch.randn(num_samples, output_dim) * 0.1


# Create a DataLoader
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate the model
hidden_dim = 64

# Define the loss function and optimizer
criterion = nn.MSELoss()

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
    parser.add_argument("--N", type=int, default=50, help="Number of Particles")

    parser.add_argument("--epo", type=int, default=10, help="Number of Epochs")
    parser.add_argument("--stop", type=int, default=1e15, help="End Epochs after this number of batches")

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

    return parser.parse_args()

def init_neptune(args):

    with open(".neptune_tok", "r") as f:
        tok = f.read()

    run = neptune.init_run(
        project="halcyon/regswarm",
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

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    if args.gradient:
        model = RegressionNet(input_dim, hidden_dim, output_dim)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
        )

        model.apply(init_weights)

    else:
        models = [RegressionNet(input_dim,hidden_dim,output_dim) for _ in range(args.N)]
        model = models[0]

        for m in models:
            m.apply(init_weights)

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
                leak=args.leak,
                lr=args.lr,
                K=args.K,
                do_momentum=args.do_momentum,
                post_process=lambda cbo: rs(cbo),
                normalize=args.normalize,
                sub_swarms=args.sub
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



    train_context = torch.no_grad if not args.gradient else contextlib.nullcontext

    if args.timeit > 0:
        time(optimizer, model, train_loader, args.timeit)

    model.train()
    # Training loop
    with train_context():
        for epoch in range(args.epo):
            for i, (x, y) in enumerate(dataloader):

                if args.gradient:
                    optimizer.zero_grad()

                    pred = model(x)
                    loss = criterion(pred, y)
                    loss.backward()

                    optimizer.step()
                else:
                    loss = optimizer.step(
                        model,
                        x,
                        lambda pred: criterion(pred, y)
                    )

            # Print the loss for every epoch
            print(f"Epoch [{epoch+1}/{args.epo}], Loss: {loss.item():.4f}")

    model.eval()
    # Test the trained model
    with torch.no_grad():
        n_test = 100
        test_inputs = torch.rand(n_test, input_dim) * 10 - 5
        test_outputs = model(test_inputs)

        expected_outputs = f(test_inputs)
        test_loss = criterion(test_outputs, expected_outputs)

        # Plot the test regression and targets
        plt.figure(figsize=(8, 6))

        print("X:", X.shape)
        print("Y:", Y.shape)
        print("test_inputs:", test_inputs.shape)
        print("test_outputs:", test_outputs.shape)

        plt.scatter(X.numpy(), Y.numpy(), color='blue', label='Training Data')
        plt.scatter(test_inputs.numpy(), test_outputs.numpy(), color='red', label='Predicted')
        plt.scatter(test_inputs.numpy(), expected_outputs.numpy(), color='green', label='Expected')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Regression Test')
        plt.legend()
        plt.show()


    if args.neptune:
        run.stop()

if __name__ == "__main__":

    args = parse_args()
    main(args)
