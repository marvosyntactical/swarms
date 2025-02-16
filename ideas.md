# Advanced Software Practical

## Experiments & Results

0. Implementation
* swarm.py, parallel and for loop implementations

1. Benchmark functions
* SGA better than CBO

2. MNIST MLP
* CBO better than SGA

3. MNIST CONV
* SGA much better, CBO collapses

4. Regression
* SGA slightly better than CBO

5. PINN
* loss increases drastically

TODO:
1. Project Writeup: Intro, Def of SGA, CBO, PSO

### TODO Gradswarm

1. Get to run on MNIST
2. Implement Spirals



## Jakob Infos

* [NOTE] kleines Netz: schwarm algos verbessern sich, adam verschlechtert sich! (wann ist es hilfreich, mit vielen partikeln ein kleines modell zu trainieren? handy?)
* [NOTE] Diffevo very good at "accelerating" downwards, not good at exploitation
* [QUES] Deadline project report?

## TODOs Project Wrap-up

* [TODO] time which algorithm is faster
* [TODO] How to improve the software product? Tests? Clean Github Repo?
* [TODO] documentation
* [TODO] code cleaning
* [TODO] write project report

* [TODO] Further experiments to run: Diffevo RL, Fix PSO


## Notes

* [NOTE] MNIST conv good SGA hyperparams: python3 mnist.py --c2 0 --do-momentum --gpu --K 5 --resample -1 --epo 1 --arch conv --lr 0.1 (normalize 2 !) (~78% hyperparam sensitive)
* [NOTE] most positive effects on bench.py and regression, emphasize these
* [NOTE] i get NaNs in CBO, SGA after ~10-50 steps with previously working hyperparams ... tuning makes this go away
* [NOTE] SGA beats CBO on mnist like this: python3 mnist.py --c2 0 --do-momentum --K 5 --normalize 2 --lr 0.01 --resample -1 (81% after 1 ep, CBO 75%)
* [NOTE] Also: python3 bench.py --c2 0 --do-momentum --objective rastrigin --K 5 (much better than CBO)
* [NOTE] Diffevo good at accelerating downwards (e.g. bench.py with --means 50000, MUCH faster than other algos; but bad at exploitation (lower means, mnist))


## Compare Small Networks

smallest possible on MNIST:

sizes=[28\*28,10] after one epoch:

Adam: 91%
SGA: 83%
CBO: 86%

--> swarm algos actually better in low width regime -> NTK analysis?
--> swarm algos better than bigger network ...? or just faster convergence?

sizes=[28\*28,100,10], after one epoch:

Adam: 96%
SGA: 78%
CBO: 87% (another run: 83%)

sizes=[28\*28,25,10], after one epoch:

Adam: 94%
SGA: 81%
CBO: 82%

sizes=[28\*28,10,10], after one epoch:

Adam: 92%
CBO: 78%
SGA: 76%




=> All Algos perform worse with smaller hidden layer

* [NOTE] Running SGA w/ sizes [784,10]



* [TODO] even smaller network (do particles help with small width regime (loss function becomes even "less convex"))
* [TODO] SGA > CBO on regression, benchmark functions ---> Retest on MNIST
* [TODO] more difficult pde, elliptic (lapl op)

* [FAIL] regression, kleines netz, 0th order
* [FAIL] Physics informed, siehe paper: https://onlinelibrary.wiley.com/doi/pdf/10.1002/gamm.202100006
* [FAIL] zeroth order starten, dann first order
* [FAIL] latent space diffusion evolution
* [TODO] schwarm basierte first order methods (see if research exists)

# PINN

* [NOTE] vmap seems to disable swarm.X.requires\_grad; and this cant be reactivated within get\_loss either
* [NOTE] tried autograd version on cluster: runs for ~10 steps, loss explodes

# Produkt

swarm.py als software part mit obigen anwendungen
* [TODO] debugge die gefailten/schlecht performenden implementationen

### Sonstige Notes

* [DONE] buch schneller!
* [TODO] angelina heinen
* [DONE] mathematical methods vorbereiten
* [DONE] finde heraus ob advanced software practical möglich


### TODO

* verify implementation of PSO etc
* parallelize particles: expand each param tensor along new particle dim and investigate
* use lr scheduler instead of handcoded schedule
* find good hyperparameters
* multiple reference particles for SwarmGrad
* fix EGICBO grad and hessian
* SGA can recover if init var is large, better than PSO it seems
* possible application: QAT - optimize in int8



