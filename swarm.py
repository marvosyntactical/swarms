import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import torch.optim as optim

from typing import Union, Iterable, Dict, Any, Callable
import random

# For Debugging
f = lambda t: (t.isnan().sum()/t.nelement()).item()

class Swarm(optim.Optimizer):
    def __init__(
            self,
            params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
            num_particles: int = 10,
            init_type: str = "gauss",
            var: float = 0.1,
        ):
        defaults = dict(num_particles=num_particles)
        super(Swarm, self).__init__(params, defaults)

        self.N = num_particles

        # N x dim(model)
        self.X = self._initialize_particles(init_type, var)

    def _initialize_particles(self, init_type="gauss", var=0.1):
        X = [[] for _ in range(self.N)]
        for param in self.param_groups[0]["params"]:

            # TODO other init distributions
            if init_type == "gauss":
                param_copies = torch.randn(self.N, *param.shape) * var + param.unsqueeze(0)

            for i in range(self.N):
                X[i].append(param_copies[i].view(-1))
        for i in range(self.N):
            X[i] = torch.cat(X[i])
        return torch.stack(X)

    def get_best(self):
        """
        Used to set the model's parameters after self.update_swarm during step.
        Has naive argmin of current_losses implementation;
        overwrite if necessary to return e.g. historical bests.
        """
        return torch.argmin(self.current_losses)

    def update_swarm(self):
        """
        The Swarm Optimizer's Logic
        """
        raise NotImplementedError


    def step(self, closure=Callable):
        with torch.no_grad():
            current_losses = []
            for i in range(self.N):

                # Swap in this particle for model parameters
                nn.utils.vector_to_parameters(self.X[i], self.param_groups[0]['params'])

                # Compute loss for this particle
                loss = closure()
                current_losses.append(loss)


            self.current_losses = torch.Tensor(current_losses)
            self.update_swarm()

            # Update the model parameters with the best particle
            best_particle_idx = self.get_best()
            nn.utils.vector_to_parameters(self.X[best_particle_idx], self.param_groups[0]['params'])

        return self.current_losses.mean()

class PSO(Swarm):
    def __init__(
            self,
            params,
            num_particles = 10,
            c1 = 2.0, # personal # TODO is this correct?
            c2 = 2.0, # global
            inertia = 0.2,
        ):
        super(PSO, self).__init__(params, num_particles)

        self.pbests_x = self.X
        self.pbests_y = torch.Tensor([float("inf") for _ in range(self.N)])
        self.V = torch.empty_like(self.X)

        self.c1 = c1 # personal
        self.c2 = c2 # global
        self.inertia = inertia

    def get_best(self):
        return torch.argmin(self.pbests_y)

    def update_swarm(self):
        for i in range(self.N):
            curr_loss = self.current_losses[i]
            if curr_loss < self.pbests_y[i]:
                self.pbests_y[i] = curr_loss
        # torch.where(
        #     self.current_losses < self.pbests_y,
        #     self.current_losses,
        #     self.pbests_y,
        #     out=self.pbests_y
        # )
        best_idx = self.get_best()

        Vpers = self.pbests_x - self.X
        Vglob = self.pbests_x[best_idx].unsqueeze(0) - self.X

        r1 = torch.rand(Vpers.shape)
        r2 = torch.rand(Vglob.shape)

        self.V = r1 * self.c1 * Vpers + r2 * self.c2 * Vglob + self.inertia * self.V
        self.X += self.V


class SwarmGrad(Swarm):
    def __init__(
            self,
            params,
            num_particles = 10,
            c1 = 6.1, # 0.08 unnormed
            c2 = 0.0,
            inertia = 0.0,
            neg_slope = 0.1,
        ):
        super(SwarmGrad, self).__init__(params, num_particles)

        self.V = torch.empty_like(self.X)

        self.perm = list(range(self.N))

        self.c1 = c1
        self.c2 = c2
        self.neg_slope = neg_slope
        self.inertia = inertia


    def update_swarm(self):


        # assign reference particles:
        random.shuffle(self.perm)

        # Unnormalized works with larger c1 (problem dependent?)
        H = self.X[self.perm] - self.X
        H /= torch.linalg.norm(H, dim=-1).unsqueeze(-1) + 1e-5
        # print(f"H%: {f(H)}")

        fdiff = self.current_losses - self.current_losses[self.perm]
        fdiff = F.leaky_relu(fdiff, negative_slope=self.neg_slope) # TODO inplace
        fdiff = fdiff.unsqueeze(-1)
        # print(f"fd%: {f(fdiff)}")

        r1 = torch.rand(*H.shape) # U[0,1]
        r2 = torch.randn(*H.shape) # N(0,1)

        Vref = r1 * self.c1 * fdiff * H
        Vrnd = r2 * self.c2
        Vinr = self.inertia * self.V

        self.V = Vref + Vrnd + Vinr
        self.X += self.V

        avgV = torch.mean(torch.linalg.norm(self.V, dim=-1))
        print(f"Avg V={avgV}")


class SwarmGradAccel(Swarm):
    def __init__(
            self,
            params,
            num_particles = 10,
            c1 = 0.02, # 0.08 unnormed
            c2 = 0., # 0.005,
            inertia = 0.9,
            beta = 0.999,
            neg_slope = 0.1,
        ):
        super(SwarmGradAccel, self).__init__(params, num_particles)

        self.V = torch.empty_like(self.X)
        self.A = torch.empty_like(self.X)

        self.perm = list(range(self.N))

        self.c1 = c1
        self.c2 = c2
        self.neg_slope = neg_slope
        self.inertia = inertia
        self.beta = beta

        self.t = 1


    def update_swarm(self):
        t = self.t

        # assign reference particles:
        random.shuffle(self.perm)

        H = self.X[self.perm] - self.X
        H /= torch.linalg.norm(H, dim=-1).unsqueeze(-1) + 1e-5

        fdiff = self.current_losses - self.current_losses[self.perm]
        F.leaky_relu(fdiff, negative_slope=self.neg_slope, inplace=True)
        fdiff.unsqueeze_(-1)

        r1 = torch.rand(*H.shape) # U[0,1]
        r2 = torch.randn(*H.shape) # N(0,1)

        Vref = fdiff * H
        Vrnd = r2 * self.c2

        # Adam like update
        V = self.inertia * self.V + (1-self.inertia) * Vref
        A = self.beta * self.A + (1-self.beta) * Vref**2

        mthat = V/(1-self.inertia**t)
        vthat = A/(1-self.beta**t)

        # learning rate schedule; should be managed by a scheduler TODO
        # schedule_weight = 1/(1-self.inertia**t)
        schedule_weight = 1

        update = schedule_weight * self.c1 * r1 * (mthat / (torch.sqrt(vthat) + 1e-6)) + Vrnd

        self.A = A
        self.V = V
        self.X += update

        avgV = torch.mean(torch.linalg.norm(update, dim=-1))
        print(f"Avg V={avgV}")
        avgA = torch.mean(torch.linalg.norm(A, dim=-1))
        print(f"Avg A={avgA}")

        self.t += 1




class CBO(Swarm):
    pass

class CBS(Swarm):
    pass


def main(args):
    ...
    return 0

if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
