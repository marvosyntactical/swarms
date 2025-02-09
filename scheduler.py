from abc import ABC, abstractmethod


class Scheduler(ABC):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.initial_lr = optimizer.lr

    @abstractmethod
    def step(self):
        pass

    def get_lr(self):
        return self.optimizer.lr


class NoScheduler(Scheduler):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass

class StepLR(Scheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch % self.step_size == 0:
            new_lr = self.get_lr() * self.gamma
            self.optimizer.set_lr(new_lr)

class ExponentialLR(Scheduler):
    def __init__(self, optimizer, gamma):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        new_lr = self.get_lr() * self.gamma
        self.optimizer.set_lr(new_lr)

class CosineAnnealingLR(Scheduler):
    def __init__(self, optimizer, T_max, eta_min=0):
        super().__init__(optimizer)

        self.T_max = T_max
        self.eta_min = eta_min
        self.current_epoch = 0

    def step(self):
        import math

        self.current_epoch += 1
        if self.current_epoch <= self.T_max:
            eta_t = self.eta_min + (self.initial_lr - self.eta_min) * (1 + math.cos(math.pi * self.current_epoch / self.T_max)) / 2
            self.optimizer.set_lr(eta_t)

class ReduceLROnPlateau(Scheduler):
    def __init__(self, optimizer, mode='min', factor=0.8, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0):
        super().__init__(optimizer)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

    def step(self, metrics):
        current = metrics
        if self.best is None:
            self.best = current
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        if (self.mode == 'min' and current < self.best - self.threshold) or \
           (self.mode == 'max' and current > self.best + self.threshold):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            print(f"=====PLATEAU SCHEDULER STEP=====")
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown
            new_lr = max(self.get_lr() * self.factor, self.min_lr)
            self.optimizer.set_lr(new_lr)
            print(f"New lr={new_lr}")
