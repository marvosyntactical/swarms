"""
Compare swarm optimisers across the benchmark functions in bench.py
and produce a benchmark figure (PNG).

Usage:
    python3 eval.py                        # default settings -> benchmark.png
    python3 eval.py --runs 5 --iterations 800 --out benchmark.png
"""
import argparse
import random
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from bench import ObjectiveFunction, ackley, rastrigin, xsy4, sphere, griewank
from swarm import PSO, CBO, EGICBO, SwarmGradAccel, DiffusionEvolution
from diffevo.fitnessmapping import Identity


OBJECTIVES = {
    "sphere":    sphere,
    "ackley":    ackley,
    "rastrigin": rastrigin,
    "griewank":  griewank,
    "xsy4":      xsy4,
}

OPTIM_COLORS = {
    "SGA":     "#d62728",
    "PSO":     "#1f77b4",
    "CBO":     "#2ca02c",
    "EGI":     "#9467bd",
    "DiffEvo": "#ff7f0e",
}


def make_optimizer(name, models, iterations, device):
    if name == "SGA":
        return SwarmGradAccel(
            models, c1=1.0, c2=0.0, beta1=0.9, beta2=0.99, lr=1.0,
            K=5, leak=0.1, do_momentum=True, normalize=2,
            sub_swarms=1, parallel=True, device=device,
        )
    if name == "PSO":
        return PSO(
            models, c1=1.5, c2=0.5, inertia=0.5,
            do_momentum=False, parallel=True, device=device,
        )
    if name == "CBO":
        return CBO(
            models, lambda_=1.0, sigma=0.5, dt=0.1, temp=30.0,
            noise_type="component", parallel=True, device=device,
        )
    if name == "EGI":
        return EGICBO(
            models, lambda_=0.9, sigma=0.5, kappa=1e5, slack=10.0,
            dt=0.1, temp=30.0, noise_type="component",
            extrapolate=False, parallel=True, device=device,
        )
    if name == "DiffEvo":
        return DiffusionEvolution(
            models, num_steps=iterations + 8, noise=1.0,
            fitness_mapping=Identity(), parallel=True, device=device,
        )
    raise ValueError(name)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single(opt_name, fn, dim, N, iterations, std, mean, seed, device):
    seed_all(seed)
    mean_t = torch.tensor([mean] * dim)
    models = [ObjectiveFunction(fn, dim, mean_t, std) for _ in range(N)]
    opt = make_optimizer(opt_name, models, iterations, device)
    model = opt.model
    dummy = torch.tensor([0.0])

    best = float("inf")
    curve = np.empty(iterations, dtype=np.float64)
    with torch.no_grad():
        for t in range(iterations):
            loss = opt.step(model, dummy, lambda x: x).item()
            best = min(best, loss)
            curve[t] = best
    return curve


def run_grid(args):
    optims = args.optims
    results = {fn_name: {} for fn_name in args.functions}
    total = len(args.functions) * len(optims) * args.runs
    done = 0
    t0 = time.time()
    for fn_name in args.functions:
        fn = OBJECTIVES[fn_name]
        for opt_name in optims:
            curves = np.empty((args.runs, args.iterations), dtype=np.float64)
            for r in range(args.runs):
                curves[r] = run_single(
                    opt_name, fn, args.dim, args.N, args.iterations,
                    args.std, args.mean, seed=r, device=args.device,
                )
                done += 1
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] {fn_name:>9s} / {opt_name:<8s} "
                      f"run {r+1}/{args.runs}  best={curves[r, -1]: .4e}  "
                      f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s")
            results[fn_name][opt_name] = curves
    return results


def plot_grid(results, args):
    fns = list(results.keys())
    n = len(fns)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.6 * rows),
                             squeeze=False)
    axes = axes.flatten()

    for i, fn_name in enumerate(fns):
        ax = axes[i]
        for opt_name, curves in results[fn_name].items():
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            x = np.arange(curves.shape[1])
            color = OPTIM_COLORS.get(opt_name, None)
            ax.plot(x, mean, label=opt_name, color=color, linewidth=1.6)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
        ax.set_title(fn_name)
        ax.set_xlabel("iteration")
        ax.set_ylabel("best loss")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.grid(True, which="both", alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Swarm optimisers on benchmark functions  "
        f"(dim={args.dim}, N={args.N}, init=N({args.mean},{args.std}^2), "
        f"{args.runs} runs, mean ± std)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=40, help="Number of particles")
    p.add_argument("--dim", type=int, default=30, help="Problem dimension")
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--runs", type=int, default=3, help="Independent seeds per cell")
    p.add_argument("--std", type=float, default=2.0, help="Init std")
    p.add_argument("--mean", type=float, default=5.0, help="Init mean (per coord)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="benchmark.png")
    p.add_argument("--functions", nargs="+", default=list(OBJECTIVES.keys()))
    p.add_argument("--optims", nargs="+",
                   default=["SGA", "PSO", "CBO", "EGI", "DiffEvo"])
    return p.parse_args()


def main():
    args = parse_args()
    results = run_grid(args)
    plot_grid(results, args)


if __name__ == "__main__":
    main()
