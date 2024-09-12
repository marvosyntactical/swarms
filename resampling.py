from typing import Callable, List
import torch

from swarm import Swarm

# Taken from CBX's https://github.com/PdIPS/CBXpy/blob/main/cbx/utils/resampling.py

def apply_resampling_default(swarm: Swarm):
    z = torch.randn(*swarm.X.shape).to(swarm.X.device)
    swarm.X += swarm.sigma * (swarm.dt**.5) * z

class resampling:
    """
    Resamplings from a list of callables

    Parameters
    ----------
    resamplings: list
        The list of resamplings to apply. Each entry should be a callable that accepts exactly one argument (the dynamic object) and returns a one-dimensional
        numpy array of indices. Makes multiple resampling criteria possible.

    apply: Callable
        - ``dyn``: The dynmaic which the resampling is applied to.
        - ``idx``: List of indices that are resampled.

        The function that should be performed on a given dynamic for selected indices. This function has to have the signature apply(dyn,idx).

    """
    def __init__(self, resamplings: List[Callable], M: int, apply: Callable = None):
        self.resamplings = resamplings
        self.M = M
        self.num_resampling = torch.zeros(M)
        self.apply = apply if apply is not None else apply_resampling_default

    def __call__(self, swarm):
        """
        Applies the resamplings to a given swarm

        Parameters
        ----------
        swarm
            The swarm object to apply resamplings to

        Returns
        -------
        None
        """
        idx = torch.unique(torch.cat([r(swarm) for r in self.resamplings]))
        if len(idx):
            print("Resampling !")
            self.apply(swarm)
            self.num_resampling[idx] += 1


class loss_update_resampling:
    """
    Resampling based on loss update difference

    Parameters
    ----------
    M: int
        The number of runs in the dynamic object the resampling is applied to.

    wait_thresh: int
        The number of iterations to wait before resampling. The default is 5. If the best loss is not updated after the specified number of
        iterations, the ensemble is resampled.

    Returns
    -------

    The indices of the runs to resample as a numpy array.
    """

    def __init__(self, M: int, wait_thresh: int = 5):
        self.M = M
        self.best_energy = float('inf') * torch.ones((self.M,))
        self.wait = torch.zeros((self.M,), dtype=int)
        self.wait_thresh = wait_thresh

    def __call__(self, swarm):
        # NOTE: M dimension/indexing dim ignored; only implemented for M=1 swarm
        self.wait += 1
        u_idx = swarm.get_best() < self.best_energy # has the swarm improved?
        self.wait[u_idx] = 0 # reset counter if swarm improved
        idx = torch.where(self.wait >= self.wait_thresh)[0] # indexes only run if it hasnt improved in wait_thresh updates

        self.wait = self.wait % self.wait_thresh # resets indexed run
        self.best_energy[u_idx] = swarm.get_best().unsqueeze(0).float()[u_idx] # update best loss if improved
        return idx
