# Advanced Software Practical

* [NOTE] i get NaNs in CBO, SGA after ~10-50 steps with previously working hyperparams ... tuning makes this go away
* [NOTE] SGA beats CBO on mnist like this: python3 mnist.py --c2 0 --do-momentum --K 5 --normalize 2 --lr 0.01 --resample -1 (81% after 1 ep, CBO 75%)
* [NOTE] Also: python3 bench.py --c2 0 --do-momentum --objective rastrigin --K 5 (much better than CBO)

## Compare Small Networks

smallest possible on MNIST: sizes=[28\*28,10] after one epoch:

Adam: 91%
SGA: 83%
CBO: 86%

--> swarm algos better than bigger network ...? or just faster convergence?

sizes=[28\*28,100,10] as before, after one epoch:

SGA: 78%



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
* [TODO] try autograd version on cluster

# Produkt

swarm.py als software part mit obigen anwendungen
* [TODO] debugge die gefailten/schlecht performenden implementationen

### Sonstige Notes

* [DONE] buch schneller!
* [TODO] angelina heinen
* [DONE] mathematical methods vorbereiten
* [DONE] finde heraus ob advanced software practical möglich
