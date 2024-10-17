import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from typing import Union, Iterable, Dict, Any, Callable, Optional
from collections import OrderedDict

import random
from scipy.sparse.linalg import svds
import numpy as np

from torch.func import vmap, functional_call, stack_module_state

# For Debugging
f = lambda t: (t.isnan().sum()/t.nelement()).item()

class Swarm(optim.Optimizer):
    def __init__(
            self,
            models: Iterable[nn.Module],
            device: torch.device = "cpu",
            post_process: Callable = lambda s: None,
        ):

        num_particles = len(models)
        self.N = num_particles

        defaults = dict(num_particles=num_particles)
        super(Swarm, self).__init__(models[0].parameters(), defaults)

        self.X = self._initialize_particles(models, device=device)
        self.pprop = self._get_pprops(models)
        self.pp = post_process

        self.pbests_x = self.X
        self.pbests_y = torch.Tensor([float("inf") for _ in range(self.N)]).to(self.X.device)

        # unnecessary:
        self.current_losses = self.pbests_y.clone()

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
        # adapted from https://github.com/PdIPS/CBXpy/blob/main/cbx/utils/torch_utils.py
        pnames = [p[0] for p in models[0].named_parameters()]
        params, buffers = stack_module_state(models)

        N = list(params.values())[-1].shape[0]
        X = torch.concatenate([params[pname].view(N,-1).detach() for pname in pnames], dim=-1)
        X = X.to(device)
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

        with torch.no_grad():

            # required = loss_fn, model, x
            # assert None not in required, required

            def get_loss(inp, X):
                pprop = self.pprop
                params = {p: X[pprop[p][-2]:pprop[p][-1]].view(pprop[p][0]) for p in pprop}
                pred = functional_call(model, (params, {}), inp)
                # loss = loss_func(pred, y) # OLD
                loss = loss_fn(pred) # NEW (RL compatible)
                return loss

            get_losses = vmap(get_loss, (None, 0))
            self.current_losses = get_losses(x, self.X)

            self.update_bests()
            self.update_swarm()
            self.pp(self)

            # Update the model (models[0]) parameters with the best particle
            best_particle_idx = self.get_best()
            nn.utils.vector_to_parameters(self.X[best_particle_idx], self.param_groups[0]['params'])

        return self.current_losses.min()

    def stats(self):
        return {}


class PSO(Swarm):
    def __init__(
            self,
            models: Iterable[nn.Module],
            c1 = 1.5, # personal # TODO is this correct?
            c2 = 0.5, # global
            inertia = 0.1,
            device: torch.device = "cpu",
            do_momentum: bool = False,
        ):
        super(PSO, self).__init__(models, device)

        self.V = torch.zeros_like(self.X)
        self.A = torch.zeros_like(self.X)

        self.c1 = c1 # personal
        self.c2 = c2 # global
        self.inertia = inertia

        # NOTE dummy, add as arguments
        self.beta1 = 0.7
        self.beta2 = 0.9
        self.t = 1
        self.lr = 1.0

        self.do_momentum = do_momentum

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
        ):
        super(SwarmGradAccel, self).__init__(models, device, post_process=post_process)

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
            # fdiffk = F.gelu(fdiffk) # NOTE TODO TEST REMOVE ME FIXME
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
        ):
        super(CBO, self).__init__(models, device, post_process=post_process)

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
        ):
        super(EGICBO, self).__init__(models, device, post_process=post_process)

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


def main(args):
    return 0

if __name__ == "__main__":
    import sys

    main(sys.argv[1:])



