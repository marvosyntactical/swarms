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

import gym
import numpy as np
from torch.distributions import Categorical

import matplotlib.pyplot as plt

# RL by Claude Opus


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


def ppo_update(model, optimizer, memory, epochs, epsilon, gamma, device, args, episode, step, run):

    states = torch.FloatTensor(np.array(memory.states)).to(device)
    actions = torch.LongTensor(memory.actions).to(device)
    rewards = torch.FloatTensor(memory.rewards).to(device)
    values = torch.FloatTensor(memory.values).to(device)
    log_probs = torch.FloatTensor(memory.log_probs).to(device)
    dones = torch.FloatTensor(memory.dones).to(device)

    returns = []
    R = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        R = reward + gamma * R * (1 - done) # reward decay
        returns.insert(0, R)
    returns = torch.tensor(returns).to(device)

    advantages = returns - values

    def loss_fn(model_out):
        action_probs, state_values = model_out

        # NOTE: Cannot init torch.distribution object because this contains if/else control flow
        # and Swarm.step uses pytorch's vmap for parallelization between models, which does not
        # allow for control flow

        # dist = Categorical(action_probs)
        # new_log_probs = dist.log_prob(actions)

        def custom_categorical_log_prob(probs, actions):
            # Ensure numerical stability by clipping probabilities
            probs = torch.clamp(probs, min=1e-8, max=1-1e-8)

            # Select the probabilities of the taken actions
            action_probs = probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

            # Compute log probabilities
            log_probs = torch.log(action_probs)

            return log_probs

        new_log_probs = custom_categorical_log_prob(action_probs, actions)

        ratio = (new_log_probs - log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

        actor_loss = - torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(state_values.squeeze(), returns)

        loss = actor_loss + 0.5 * critic_loss

        return loss

    for e in range(epochs):
        model.train()

        if args.gradient:
            loss = loss_fn(model(states))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss = optimizer.step(
                model,
                states,
                loss_fn
            )

            if args.neptune:
                for stat, val in optimizer.stats().items():
                    run[f"train/{stat}"].append(val)

        # print(f"Train Episode | Step | Epoch | Loss : {episode}\t{step}\t{e}\t{loss:.6f}")

        if args.neptune:
            run["train/loss"].append(loss.item())



def train_ppo(env, model, optimizer, num_episodes, update_interval, epochs, epsilon, gamma, device, args, run):

    memory = PPOMemory()

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        step = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs, value = model(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _, _ = env.step(action.item())
            total_reward += reward

            memory.add(
                state.cpu().numpy().flatten(),
                action.item(),
                reward,
                value.item(),
                log_prob.item(),
                done,
            )

            if len(memory.states) >= update_interval or done:
                ppo_update(
                    model,
                    optimizer,
                    memory,
                    epochs,
                    epsilon,
                    gamma,
                    device,
                    args,
                    episode,
                    step,
                    run
                )
                memory.clear()

            state = next_state
            step += 1

        print(f"Episode {episode}, Total Reward: {total_reward}, Steps: {step}")

        if episode and episode % 50 == 0:
            for testrun in range(0): # NOTE deactivate viz
                reward = visualize_episode(env, model, device)
                print(f"Visualized Test Run {testrun}, Total Reward {reward}")

        if args.neptune:
            run["train/total_reward"].append(total_reward)

def visualize_episode(env, model, device):
    state, _ = env.reset()
    done = False
    total_reward = 0

    plt.figure(figsize=(8, 6))
    img_plot = plt.imshow(env.render())
    plt.axis('off')

    while not done:
        plt.pause(0.01)
        # print("Renderin'")
        # arr = env.render() # This line renders the environment
        # print(arr.shape)

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, _ = model(state)

        action = torch.argmax(action_probs).item()

        state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Update the image
        img_plot.set_data(env.render())
        plt.draw()

    plt.close()
    env.close()
    return total_reward

def parse_args():
    parser = argparse.ArgumentParser(description="Swarm Experiments on MNIST")
    parser.add_argument("--env", type=str, default='CartPole-v1', help="OpenAI Gym Environment Name")

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

    parser.add_argument("--episodes", type=int, default=1000, help="Number of Episodes")
    parser.add_argument("--epochs", type=int, default=4, help="Number of Epochs")

    parser.add_argument("--neptune", action="store_true", help="Log to Neptune?")

    # ==== Optimizer Specific Hyperparameters ====

    parser.add_argument("--c1", type=float, default=1.0, help="c1 Hyperparameter of optimizer")
    parser.add_argument("--c2", type=float, default=1.0, help="c2 Hyperparameter of optimizer")
    parser.add_argument("--inertia", type=float, default=0.1, help="Inertia Hyperparam")
    parser.add_argument("--resample", type=int, default=40, help="Resample if swarm has not improved for this many updates. Negative for no resampling.")

    # Momentum related Hyperparameters
    parser.add_argument("--do-momentum", action="store_true", help="Whether to use momentum update")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 Hyperparameter of Momentum")
    parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 Hyperparameter of Momentum")

    # SGA
    parser.add_argument("--lr", type=float, default=1.0, help="Optional learning rate of SGA")
    parser.add_argument("--K", type=int, default=1, help="# Reference particles of SGA")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize drift")
    parser.add_argument("--sub-swarms", type=int, default=1, help="Number of sub swarms (flat hierachy)")

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
        project="halcyon/rlswarm",
        api_token=tok,
    )

    run["parameters/gradient"] = args.gradient
    run["parameters/optim"] = args.optim
    run["parameters/N"] = args.N
    run["parameters/epochs"] = args.epochs

    return run



def main(args):

    if args.neptune:
        run = init_neptune(args)
    else:
        run = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the environment and model
    env = gym.make(args.env, render_mode="rgb_array")

    print(env.observation_space.shape)

    input_dim = env.observation_space.shape[0]
    try:
        n_actions = env.action_space.n
    except AttributeError:
        n_actions = env.action_space.shape[0]


    if args.gradient:
        model = ActorCritic(input_dim, n_actions).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
        )

    else:
        models = [ActorCritic(input_dim, n_actions).to(device) for _ in range(args.N)]
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
                sub_swarms=args.sub_swarms
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
            run["parameters/sub_swarms"] = args.sub_swarms

        elif opt == "gsa":
            optimizer0 = GradSwarm(
                models,
                device=device,
                opt_args={"lr": args.lr},
            )
        elif opt == "pla":
            optimizer = PlanarSwarm(
                models,
            )
        else:
            raise NotImplementedError(f"Optim={opt}")

    # Hyperparameters TODO make argparse args
    num_episodes = args.episodes
    epochs = args.epochs

    # NOTE: try not to fiddle with hyperparams that are relatively
    # orthogonal to the kind of optimizer used

    update_interval = 128
    epsilon = 0.02
    gamma = 0.99

    #  ----- Train the model -----

    # Dont compute gradients in case of Swarm optimizer
    train_context = torch.no_grad if not args.gradient else contextlib.nullcontext

    with train_context():
        # Run the training
        train_ppo(env, model, optimizer, num_episodes, update_interval, epochs, epsilon, gamma, device, args, run)

    if args.neptune:
        run.stop()


if __name__ == "__main__":

    args = parse_args()
    main(args)
