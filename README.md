# Swarm Optimizers in PyTorch

Implementations of PSO, SwarmGrad (with Acceleration) and more to come.


### TODO

* verify implementation of PSO etc
* parallelize particles: expand each param tensor along new particle dim and investigate
* use lr scheduler instead of handcoded schedule
* find good hyperparameters
* multiple reference particles for SwarmGrad
* fix EGICBO grad and hessian
* SGA can recover if init var is large, better than PSO it seems
* possible application: QAT - optimize in int8
