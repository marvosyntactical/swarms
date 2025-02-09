# Swarm Optimizers in PyTorch

This Repository, specifically `swarm.py`, contains implementations of 

1. SwarmGrad (novel optimizer)
2. Particle Swarm Optimization
3. Consensus Based Optimization
4. Diffusion Evolution
5. EGICBO

They are evaluated on the following tasks:

1. Benchmark Optimization Functions (`bench.py`)
2. MNIST Digit Classification (`mnist.py`)
3. Regression (`reg.py`)
4. Physics Informed Neural Networks (`burger.py`)
5. Language Modeling (`gpt/gpt\_train.py`)


## References

PSO:
```
@inproceedings{eberhart1995particle,
    title={Particle swarm optimization},
    author={Eberhart, Russell and Kennedy, James},
    booktitle={Proceedings of the IEEE international conference on neural networks},
    volume={4},
    pages={1942--1948},
    year={1995},
    organization={Citeseer}
}
```

CBO:
```
@article{pinnau2017consensus,
    title={A consensus-based model for global optimization and its mean-field limit},
    author={Pinnau, Ren{\'e} and Totzeck, Claudia and Tse, Oliver and Martin, Stephan},
    journal={Mathematical Models and Methods in Applied Sciences},
    volume={27},
    number={01},
    pages={183--204},
    year={2017},
    publisher={World Scientific}
}
```

EGICBO:
```
@article{schillings2023ensemble,
    title={Ensemble-based gradient inference for particle methods in optimization and sampling},
    author={Schillings, Claudia and Totzeck, Claudia and Wacker, Philipp},
    journal={SIAM/ASA Journal on Uncertainty Quantification},
    volume={11},
    number={3},
    pages={757--787},
    year={2023},
    publisher={SIAM}
}
```

Diffusion Evolution:
``` 
@article{zhang2024diffusion,
    title={Diffusion Models are Evolutionary Algorithms},
    author={Zhang, Yanbo and Hartl, Benedikt and Hazan, Hananel and Levin, Michael},
    journal={arXiv preprint arXiv:2410.02543},
    year={2024}
}
```
