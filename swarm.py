import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from typing import Union, Iterable, Dict, Any, Callable, Optional
from collections import OrderedDict
import contextlib

import random
from scipy.sparse.linalg import svds
import numpy as np

from torch.func import vmap, functional_call, stack_module_state
from torch.nn.utils.parametrize import register_parametrization

from copy import deepcopy

# for diffusion evolution
from diffevo.fitnessmapping import *
from diffevo.generator import *
from diffevo.kde import *
from diffevo.ddim import DDIMScheduler
from diffevo.latent import RandomProjection

# for grad swarm
import cvxpy as cp


# For Debugging
f = lambda t: (t.isnan().sum()/t.nelement()).item()

class Swarm(optim.Optimizer):
    def __init__(
            self,
            models: Iterable[nn.Module],
            device: torch.device = "cpu",
            post_process: Callable = lambda s: None,
            parallel: bool = True
        ):
        self.device = device

        num_particles = len(models)
        self.N = num_particles

        defaults = dict(num_particles=num_particles)

        self.parallel = parallel

        self.X = self._initialize_particles(models, device=device)

        self.model = deepcopy(models[0])

        if parallel:
            self.pprop = self._get_pprops(models)

        super(Swarm, self).__init__(self.model.parameters(), defaults)

        self.pp = post_process

        self.pbests_x = self.X
        self.pbests_y = torch.Tensor([float("inf") for _ in range(self.N)]).to(self.X.device)

        # unnecessary:
        self.current_losses = self.pbests_y.clone()

        # TODO NOTE FIXME: For PINN
        # self.X.requires_grad_(True)

    def set_lr(self, new_lr):
        if not hasattr(self, "lr"):
            raise AttributeError(f"This Optimizer does not have a .lr attribute")
        else:
            self.lr = new_lr

    def update_bests(self):
        # NOTE: This could be done with torch.where
        for i in range(self.N):
            curr_loss = self.current_losses[i]
            if curr_loss < self.pbests_y[i]:
                self.pbests_y[i] = curr_loss
                self.pbests_x[i] = self.X[i]


    def _initialize_particles(self, models: Iterable[nn.Module], device: torch.device="cpu"):
        if self.parallel:
            # adapted from https://github.com/PdIPS/CBXpy/blob/main/cbx/utils/torch_utils.py
            pnames = [p[0] for p in models[0].named_parameters()]
            params, buffers = stack_module_state(models)

            N = list(params.values())[-1].shape[0]
            X = torch.concatenate([params[pname].view(N,-1).detach() for pname in pnames], dim=-1)
            X = X.to(device)
            return X
        else:
            X = [[] for _ in range(self.N)]
            for i, m in enumerate(models):
                X[i] = nn.utils.parameters_to_vector(m.parameters())
            X = torch.stack(X).to(device)
            return X

    def _get_pprops(self, models: Iterable[nn.Module]):
        # taken from https://github.com/PdIPS/CBXpy/blob/main/cbx/utils/torch_utils.py
        params, buffers = stack_module_state(models)
        pnames = [p[0] for p in models[0].named_parameters()]
        pprop = OrderedDict()
        for p in pnames:
            a = 0
            if len(pprop)>0:
                a = pprop[next(reversed(pprop))][-1]
            pprop[p] = (params[p][0,...].shape, a, a + params[p][0,...].numel())
        return pprop


    def get_best(self):
        """
        Used to set the model's parameters after self.update_swarm during step.
        Has naive argmin of current_losses implementation;
        overwrite if necessary to return e.g. historical bests.
        """
        return torch.argmin(self.current_losses)

    def update_swarm(self):
        """
        The Swarm Optimizer's Logic. Updates self.X based on self.current_losses.
        """
        raise NotImplementedError(f"This must be implemented by classes inheriting from Swarm.")


    def step(
            self,
            model: nn.Module,
            x: torch.Tensor,
            loss_fn: Callable, # must have signature model_output -> loss
        ):

        ctx = torch.no_grad if not self.parallel else contextlib.nullcontext

        with ctx():

            if self.parallel:
                def get_loss(inp, X):
                    # print(f"inp: {inp.requires_grad}")
                    # print(f"X: {X.requires_grad}")
                    pprop = self.pprop
                    params = {p: X[pprop[p][-2]:pprop[p][-1]].view(pprop[p][0]) for p in pprop}
                    # Not allowed, requires_grad_ must be done outside vmap ...!
                    # for n, param in params.items():
                    #     param.requires_grad_(True)
                    pred = functional_call(model, (params, {}), inp)
                    # print(f"pred: {pred.requires_grad}")
                    # loss = loss_func(pred, y) # OLD
                    loss = loss_fn(pred) # NEW (RL compatible)
                    return loss

                # self.X.requires_grad_(True)
                get_losses = vmap(get_loss, (None, 0))
                self.current_losses = get_losses(x, self.X)

            else:
                for i in range(self.N):
                    # TODO: Replace self.model weights with weights given by param i
                    # print(f"Substituting vector {i}")
                    nn.utils.vector_to_parameters(self.X[i], self.model.parameters())
                    # print(f"Forwarding model with vector {i}")
                    pred = self.model(x)
                    # print(f"Forwarding loss with vector {i}")
                    loss = loss_fn(pred)
                    self.current_losses[i] = loss

            self.update_bests()
            self.update_swarm()
            self.pp(self)

            # Update the model (models[0]) parameters with the best particle
            best_particle_idx = self.get_best()
            nn.utils.vector_to_parameters(self.X[best_particle_idx], self.param_groups[0]['params'])

        return self.current_losses.min()

    def stats(self):
        return {}


class DiffusionEvolution(Swarm):
    def __init__(
            self,
            models: Iterable[nn.Module],
            device: torch.device = "cpu",
            num_steps: int = 100,
            noise: float = 1.0,
            fitness_mapping=None,
            latent_dim: Optional[int] = None,
            parallel: bool = True,
        ):
        # adapted from https://github.com/Zhangyanbo/diffusion-evolution/tree/096b1b267f957905d6b9aea1d3f2866eebbd6d65

        super(DiffusionEvolution, self).__init__(models, device, parallel=parallel)

        self.num_steps = num_steps
        self.noise = noise

        if fitness_mapping is None:
            self.fitness_mapping = Identity()
        else:
            self.fitness_mapping = fitness_mapping

        self.scheduler = DDIMScheduler(self.num_steps)

        if latent_dim is not None:
            self.random_map = RandomProjection(self.X.shape[-1], latent_dim,
                    normalize=True).to(device)

    def update_swarm(self):
        _, alpha = next(self.scheduler)
        f = self.fitness_mapping(self.current_losses)

        if hasattr(self, "random_map"):
            G = LatentBayesianGenerator(self.X, self.random_map(self.X).detach(), f, alpha, density="kde")
        else:
            G = BayesianGenerator(self.X, f, alpha, density="kde")

        self.X = G(noise=self.noise)

    def stats(self):
        # TODO
        return {}




class PSO(Swarm):
    def __init__(
            self,
            models: Iterable[nn.Module],
            c1 = 1.5, # personal # TODO is this correct?
            c2 = 0.5, # global
            beta1 = 0.9,
            beta2 = 0.99,
            inertia = 0.1,
            device: torch.device = "cpu",
            post_process: Callable = lambda s: None,
            do_momentum: bool = False,
            parallel: bool = True,
        ):
        super(PSO, self).__init__(models, device, post_process=post_process, parallel=parallel)

        self.V = torch.zeros_like(self.X)
        self.A = torch.zeros_like(self.X)

        self.c1 = c1 # personal
        self.c2 = c2 # global
        self.inertia = inertia

        # NOTE dummy, add as arguments
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 1
        self.lr = 1.0

        self.do_momentum = do_momentum

        # NOTE FIXME: hardcoded for access by resampling
        self.sigma = 0.1
        self.dt = 0.1

    def get_best(self):
        return torch.argmin(self.pbests_y)

    def update_swarm(self):
        t = self.t
        best_idx = self.get_best()

        Vpers = self.pbests_x - self.X
        Vglob = self.pbests_x[best_idx].unsqueeze(0) - self.X

        r1 = torch.rand(*Vpers.shape).to(self.X.device)
        r2 = torch.rand(*Vglob.shape).to(self.X.device)

        Vcurr = r1 * self.c1 * Vpers + r2 * self.c2 * Vglob

        if self.do_momentum:

            V = self.beta1 * self.V + (1-self.beta1) * Vcurr
            A = self.beta2 * self.A + (1-self.beta2) * Vcurr**2

            mthat = V/(1-self.beta1**t)
            vthat = A/(1-self.beta2**t)

            update = self.lr * (mthat / (torch.sqrt(vthat) + 1e-9))

            self.A = A
            self.V = V
            self.X += update

            self.t += 1
        else:
            V = Vcurr + self.inertia * self.V
            self.V = V
            self.X += self.V


    def stats(self):
        avgV = torch.mean(torch.linalg.norm(self.V, dim=-1)).item()

        return {"avg_v_norm": avgV}


class SwarmGradAccel(Swarm):
    def __init__(
            self,
            models,
            c1 = 1., # 0.08 unnormed
            c2 = 0., # 0.005,
            beta1 = 0.9,
            beta2 = 0.99,
            leak = 0.1,
            lr = 1.0,
            K = 1,
            device: torch.device = "cpu",
            do_momentum: bool = True,
            post_process: Callable = lambda s: None,
            normalize: int = 0,
            sub_swarms: int = 1,
            parallel: bool = True,
        ):
        super(SwarmGradAccel, self).__init__(models, device, post_process=post_process, parallel=parallel)

        self.V = torch.zeros_like(self.X)
        self.A = torch.zeros_like(self.X)

        # single swarm case:
        self.perms = [list(range(self.N)) for _ in range(K)]

        # sub swarms case:
        assert self.N % sub_swarms == 0, (self.N, sub_swarms)
        self.ss_size = ss_size = int(self.N/sub_swarms)
        self.sub_perms = [[list(range(ss*ss_size, (ss+1)*ss_size)) for ss in range(sub_swarms)] for _ in range(K)]
        self.sub_swarms = sub_swarms
        self.merged = False

        self.c1 = c1
        self.c2 = c2
        self.leak = leak
        self.beta1 = beta1
        self.beta2 = beta2
        self.K = K

        self.lr = lr

        self.t = 1
        self.do_momentum = do_momentum
        self.normalize = normalize

        # NOTE FIXME: hardcoded for access by resampling
        self.sigma = 0.1
        self.dt = 0.1

    def assign_ref(self, k: int):

        if self.sub_swarms == 1 or self.merged:
            random.shuffle(self.perms[k])
            ref = self.perms[k]
        else:
            ref = []
            for sub in range(self.sub_swarms):
                random.shuffle(self.sub_perms[k][sub])
                ref += self.sub_perms[k][sub]

        return ref


    def update_swarm(self):
        t = self.t
        if t == 1000:
            # print("="*30)
            print("SUB SWARMS MERGED")
            self.merged = True

        Vref = torch.zeros_like(self.X)

        for k in range(self.K):
            # assign kth new reference particle:
            ref_k = self.assign_ref(k)

            # vector pointing to reference particle
            Hk = self.X[ref_k] - self.X
            if self.normalize:
                Hk /= torch.linalg.norm(Hk, dim=-1).unsqueeze(-1)**self.normalize + 1e-9

            # difference in loss 
            fdiffk = self.current_losses - self.current_losses[ref_k]
            # leaky relu decreases difference if particle is already better than reference
            # so we dont move too much in the opposite direction
            F.leaky_relu(fdiffk, negative_slope=self.leak, inplace=True)
            fdiffk.unsqueeze_(-1)

            Vref += fdiffk * Hk

        Vref /= self.K

        r1 = 1.0 # NOTE TODO TEST REMOVE
        # r1 = torch.rand(*self.X.shape).to(self.X.device) # U[0,1]
        r2 = torch.randn(*self.X.shape).to(self.X.device) # N(0,1)

        Vrnd = r2 * self.c2

        if self.do_momentum:
            # Adam like update

            V = self.beta1 * self.V + (1-self.beta1) * Vref
            A = self.beta2 * self.A + (1-self.beta2) * Vref**2

            mthat = V/(1-self.beta1**t)
            vthat = A/(1-self.beta2**t)

            update = self.lr * self.c1 * r1 * (mthat / (torch.sqrt(vthat) + 1e-9)) + Vrnd

            self.A = A
            self.V = V
            self.X += update

        else:
            V = self.c1 * r1 * Vref + Vrnd
            self.V = V
            self.X += V

        self.t += 1


    def stats(self):
        avgV = torch.mean(torch.linalg.norm(self.V, dim=-1)).item()
        avgA = torch.mean(torch.linalg.norm(self.A, dim=-1)).item()

        return {"avg_v_norm": avgV, "avg_a_norm": avgA}


class CBO(Swarm):
    def __init__(
            self,
            models: Iterable[nn.Module],
            lambda_ = 1.5, # drift
            sigma = 0.8, # diff
            temp = 50.0,
            dt = 0.1,
            noise_type = "component",
            device: torch.device = "cpu",
            post_process: Callable = lambda s: None,
            do_momentum: bool = False,
            parallel: bool = True
        ):
        super(CBO, self).__init__(models, device, post_process=post_process, parallel=parallel)

        self.lambda_ = lambda_ # drift
        self.sigma = sigma # diff
        self.temp = temp
        self.dt = dt

        self.V = torch.zeros_like(self.X)
        self.A = torch.zeros_like(self.X)

        self.noise_type = noise_type
        self.do_momentum = do_momentum

        # NOTE dummy, add as arguments
        self.beta1 = 0.7
        self.beta2 = 0.9
        self.t = 1
        self.lr = 1.0

    def get_softmax(self):
        # logsumexp trick from https://github.com/PdIPS/CBXpy/blob/main/cbx/utils/torch_utils.py
        weights = - (self.temp * self.current_losses).to(self.X.device)
        coeffs = torch.exp(weights - torch.logsumexp(weights, dim=-1, keepdims=True))
        return (self.X * coeffs.unsqueeze(1)).sum(axis=0)

    def update_swarm(self):
        t = self.t
        softmax = self.get_softmax()

        Vdet = softmax.unsqueeze(0) - self.X

        B = torch.randn(*Vdet.shape).to(self.X.device)

        if self.noise_type == "component":
            # componentwise / anisotropic
            diffusion = Vdet * B
        else:
            # proportional to norm / isotropic
            diffusion = torch.linalg.norm(Vdet, dim=-1).unsqueeze(1) * B

        Vrnd = self.sigma * (self.dt**.5) * diffusion

        if self.do_momentum:
            # CBO w/ momentum: https://arxiv.org/abs/2012.04827

            V = self.beta1 * self.V + (1-self.beta1) * Vdet
            A = self.beta2 * self.A + (1-self.beta2) * Vdet**2

            mthat = V/(1-self.beta1**t)
            vthat = A/(1-self.beta2**t)

            update = self.lr * self.lambda_ * self.dt * (mthat / (torch.sqrt(vthat) + 1e-9)) + Vrnd

            self.A = A
            self.V = V
            self.X += update
            self.t += 1
        else:
            V = self.lambda_ * self.dt * Vdet + Vrnd
            self.V = V
            self.X += self.V

    def stats(self):
        avgV = torch.mean(torch.linalg.norm(self.V, dim=-1)).item()

        return {"avg_v_norm": avgV}


class EGICBO(Swarm):
    def __init__(
            self,
            models: Iterable[nn.Module],
            lambda_ = 0.9, # drift
            sigma = 0.5, # diff
            kappa = 100000.0,
            slack = 10,
            dt = 0.1,
            temp = 30.0,
            extrapolate = False, # Use Hessian?
            noise_type = "component",
            post_process: Callable = lambda s: None,
            device: torch.device = "cpu",
            do_momentum: bool = False,
            parallel: bool = True
        ):
        super(EGICBO, self).__init__(models, device, post_process=post_process, parallel=parallel)

        self.lambda_ = lambda_ # drift
        self.sigma = sigma # diff
        self.temp = temp
        self.kappa = kappa
        self.slack = slack
        self.dt = dt

        self.extrapolate = extrapolate

        self.noise_type = noise_type

        # first particle serves as mean
        self.X[0] = torch.mean(self.X[1:], dim=0)

        # NOTE dummy, add as arguments
        self.beta1 = 0.7
        self.beta2 = 0.9
        self.t = 1
        self.do_momentum = do_momentum


    # def get_softmax(self):
    #     # NAIVE
    #     weights = torch.Tensor([torch.exp(-self.temp * self.current_losses[i]) for i in range(self.N)]).to(self.X.device)
    #     denominator = torch.sum(weights)
    #     num = torch.zeros_like(self.X[0])
    #     for i in range(self.N):
    #         addition = self.X[i] * weights[i]
    #         num += addition
    #     return num/(denominator + 1e-6)

    def get_softmax(self):
        # logsumexp trick from https://github.com/PdIPS/CBXpy/blob/main/cbx/utils/torch_utils.py
        weights = - (self.temp * self.current_losses).to(self.X.device)
        coeffs = torch.exp(weights - torch.logsumexp(weights, dim=-1, keepdims=True))
        return (self.X * coeffs.unsqueeze(1)).sum(axis=0)

    def grad_and_hessian(self, idx=0):
        """
        adapted from official implementation at
        https://github.com/MercuryBench/ensemble-based-gradient/blob/main/grad_inference.py
        """

        # FIXME TODO NOTE something is wrong with the Gradient calc
        X = self.X - self.X[idx].unsqueeze(0)
        Y = self.current_losses - self.current_losses[idx]

        Z = X.clone()
        Xnorms_short = torch.linalg.norm(X, dim=-1)
        Xnorms = Xnorms_short.unsqueeze(-1).expand(X.shape)
        torch.where(
            Xnorms != 0,
            X/Xnorms,
            torch.zeros_like(X),
            out=Z
        )
        mat1 = X @ Z.T # FIXME this is correctly N x N, but in Alg 1, Line 4 they imply d x N?

        hadamard = .5 * mat1**2
        mat = torch.cat([mat1, hadamard], dim=-1)

        gamma_base = 0.5**2*Xnorms_short**3

        b = torch.zeros_like(Y)
        divisor_b = gamma_base + self.slack
        torch.where(
            divisor_b != 0,
            Y/divisor_b,
            b,
            out=b
        )

        A = torch.zeros_like(mat)
        divisor_A = (torch.tile(gamma_base, (2*self.N, 1)) + self.slack).T
        torch.where(
            divisor_A != 0,
            mat/divisor_A,
            A,
            out=A,
        )

        # Least Squares
        u = torch.linalg.lstsq(A,b)[0] # NOTE: Try scipy lsmr, faster apparently?

        assert u.shape[0] == 2 * self.N, u.shape

        G = Z.T @ u[0:self.N]

        if self.extrapolate:
            coeffs_hess = u[self.N:]

            # Cannot allocate memory:
            # print([t.shape for t in [coeffs_hess, X]])
            # NOTE: This is too memory inefficient:
            # H = torch.einsum('i,ki,li->kl', coeffs_hess, X.T, X.T)

            # NOTE: This is too computationally expensive as well
            # TODO: Block diagonal apprxn of hessian, i.e. layerwise
            hu = torch.einsum('i,ki,li,ji->kj', coeffs_hess, Z.T, Z.T, X.T)

            # H = coeffs_hess * mat1 # this is wrong?
        else:
            hu =  ...

        return G, hu


    def update_swarm(self):

        G, hu = self.grad_and_hessian(idx=0)

        g = G.unsqueeze(0)
        if self.extrapolate:
            # g += H @ (self.X-self.X[0].unsqueeze(0))
            g += hu

        softmax = self.get_softmax()

        drift = softmax.unsqueeze(0) - self.X

        B = torch.randn(*drift.shape).to(self.X.device)

        if self.noise_type == "component":
            # componentwise
            diffusion = drift * B
        else:
            # proportional to norm
            diffusion = torch.linalg.norm(drift,dim=-1).unsqueeze(1) * B


        # NOTE figure out if applying lambda here is problematic
        # if lambda != 1.0
        Vdet = self.lambda_ * drift + self.kappa * g

        Vrnd = (self.dt**.5) * self.sigma * diffusion

        if self.do_momentum:
            # CBO w/ momentum: https://arxiv.org/abs/2012.04827
            V = self.beta1 * self.V + (1-self.beta1) * Vdet
            A = self.beta2 * self.A + (1-self.beta2) * Vdet**2

            mthat = V/(1-self.beta1**t)
            vthat = A/(1-self.beta2**t)

            update = self.dt * (mthat / (torch.sqrt(vthat) + 1e-9)) + Vrnd

            self.A = A
            self.V = V
            self.X += update
            self.t += 1
        else:
            V = self.dt * Vdet + Vrnd
            self.V = V
            self.X += self.V

        self.X += V

        # first particle serves as mean
        self.X[0] = torch.mean(self.X[1:], dim=0)

        # save these for stats
        self.V = V
        self.Gnorm = torch.linalg.norm(G).item()


    def stats(self):
        avgV = torch.mean(torch.linalg.norm(self.V, dim=-1)).item()
        return {
            "avg_v_norm": avgV,
            "g_norm": self.Gnorm
        }

class PlanarSwarm(Swarm):
    def __init__(
            self,
            models: Iterable[nn.Module],
            device: torch.device = "cpu"
        ):
        super(PlanarSwarm, self).__init__(models, device)


    def update_swarm(self):
        U, s, Vt = svds(self.X.numpy(), k=self.N-1)

        idx = np.argsort(s)
        s = s[idx]
        Vt = Vt[idx]

        down = Vt[-1]
        down /= np.linalg.norm(down)

        V = 10* torch.Tensor(down).to(self.X.device).unsqueeze(0)

        self.V = V
        self.X += V

class GradSwarm:
    """
    Calculates gradients at each particle using adam during a warmup phase,
    then uses current positions and current adam updates to
    arrive at estimate of point that all updates point to using SOCP.
    """
    def __init__(
            self,
            models: Iterable[nn.Module],
            device: torch.device = "cpu",
            warmup: int = 100,
            opt_cls = torch.optim.Adam,
            opt_args = {"lr": 0.01},
        ):

        self.models = models

        self.model = deepcopy(models[0])

        self.N = len(self.models)

        self.opts = [
            opt_cls(self.models[i].parameters(), **opt_args) for i in range(self.N)
        ]

        self.set_opt = lambda: opt_cls(self.model.parameters(), **opt_args)

        self.device = device

        self.warmup = warmup
        self.t = 0

    def step(
            self,
            _,
            x: torch.Tensor,
            loss_fn: Callable, # must have signature model_output -> loss
        ):
        """
        Note: Has same signature as Swarm.step, but ignores model argument
        """

        def opt_step(opt, model, x, loss_fn):

            opt.zero_grad()

            output = model(x)
            loss = loss_fn(output)

            loss.backward()

            opt.step()
            return loss

        if self.t < self.warmup:

            losses = []
            for i in range(self.N):
                opt = self.opts[i]
                m = self.models[i]

                loss = opt_step(opt, m, x, loss_fn)
                losses.append(loss)

            r = min(losses)

        else:
            if self.t == self.warmup:
                print(f"==== Approximating target point using SOCP ... ====")
                # calculate point that is closest to what current positions and gradients point to

                X = []
                V = []

                for i in range(self.N):

                    m = self.models[i]
                    opt = self.opts[i]

                    p = _get_flat_params(m)
                    # print("Parameter shape before:", p.shape)

                    v = _get_flat_adam_update_direction(opt)
                    # normalize v
                    v /= torch.linalg.norm(v)

                    X.append(p)
                    V.append(v)

                X = torch.stack(X)
                V = torch.stack(V)

                q = torch.Tensor(solve_socp(X, V)[0]).to(device=self.device)
                # print("Parameter shape after:", q.shape)

                nn.utils.vector_to_parameters(
                    q,
                    self.model.parameters()
                )

                self.opt = self.set_opt()

                # print("Shapes:", x.shape, self.model.linears[0].weight.shape)

            r = opt_step(
                self.opt,
                self.model,
                x,
                loss_fn
            )

        self.t += 1
        return r


def solve_socp(X_torch, V_torch):
    """
    Solve the SOCP:
       minimize    r
       subject to  ||(I - v_i v_i^T)(q - x_i)|| <= r   and   (q - x_i)^T v_i >= 0 for all i,
    where q is restricted to lie in the affine subspace spanned by {x_i} and {v_i}.

    We represent q as q = q0 + Bz, with q0 chosen as X[0] and B an orthonormal basis for
    the span of {x_i - q0, v_i}.

    Args:
        X_torch: PyTorch tensor of shape (N, d) for starting points.
        V_torch: PyTorch tensor of shape (N, d) for unit direction vectors.

    Returns:
        q_opt: Optimal point in R^d (numpy array of shape (d,))
        r_opt: Optimal maximum distance (scalar)
    """

    # Convert tensors to numpy arrays.
    X = X_torch.detach().cpu().numpy()  # shape (N, d)
    V = V_torch.detach().cpu().numpy()  # shape (N, d)
    N, d = X.shape

    # Choose an anchor point q0. Here, we simply pick the first point.
    q0 = X[0]

    # Construct the matrix whose columns are (x_i - q0) and v_i.
    # Y will be of shape (d, 2N).
    Y = np.hstack([ (X - q0).T, V.T ])

    # Compute an orthonormal basis for the column space of Y using SVD.
    U, S, _ = np.linalg.svd(Y, full_matrices=False)
    tol = 1e-8
    k = np.sum(S > tol)
    print(f"For a tolerance of {tol}, got a {k}-dim subspace.")
    B = U[:, :k]  # B is (d, k)

    # We now express q as: q = q0 + B * z, where z is in R^k.
    # Create CVXPY variable for z and scalar r.
    z = cp.Variable(k)
    r = cp.Variable()

    # Define q as an affine expression in z.
    q_expr = q0 + B @ z  # CVXPY understands this since q0 and B are constants.

    # Build the constraints.
    constraints = []
    for i in range(N):
        xi = X[i]
        vi = V[i]

        # Directional constraint: (q - xi)^T vi >= 0
        constraints.append( (q_expr - xi) @ vi >= 0 )

        # Orthogonal distance constraint:
        # Compute the projection of (q - xi) onto vi:  (q - xi)^T vi.
        dot_val = (q_expr - xi) @ vi
        # The orthogonal component is: (q - xi) - (dot_val)*vi.
        constraints.append( cp.norm((q_expr - xi) - dot_val * vi) <= r )

    # Objective: minimize r.
    objective = cp.Minimize(r)
    prob = cp.Problem(objective, constraints)

    print("Solving ...")
    prob.solve(solver=cp.SCS)
    print("Solved!")

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError("Optimization did not converge!")

    # Compute optimal q in the original space.
    q_opt = q0 + B @ z.value
    return q_opt, r.value

def _get_flat_params(model):
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def _get_flat_adam_update_direction(optimizer):
    updates = []
    for group in optimizer.param_groups:
        lr = group['lr']
        eps = group.get('eps', 1e-8)
        beta1, beta2 = group['betas']
        for p in group['params']:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            # Make sure state has been initialized (after at least one optimizer.step())
            if 'exp_avg' in state and 'exp_avg_sq' in state:
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                # Get the step count; default to 1 if not set.
                step = state.get('step', 1)
                # Compute bias corrections:
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                # Compute bias-corrected estimates
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                # Adam update direction for this parameter
                update = lr * m_hat / (torch.sqrt(v_hat) + eps)
                updates.append(update.view(-1))
            else:
                # If state is not yet available, fall back to the raw gradient.
                updates.append(p.grad.detach().view(-1))
    return torch.cat(updates)



def main(args):
    return 0

if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

