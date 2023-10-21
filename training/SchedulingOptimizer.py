import logging
from typing import Iterable, List, Union, Iterator

import torch
from torch import nn


class SchedulingOptimizer:
    def __init__(self, parameters: Union[Iterator[nn.Parameter], List[dict]], lr: float, total_steps: int,
                 warmup_fraction: float = 0):
        num_warmup_steps = round(total_steps * warmup_fraction)
        warned_step: bool = False

        def _get_lr(step: int):
            nonlocal warned_step
            if step < num_warmup_steps:
                current_factor = step / num_warmup_steps
            else:
                current_factor = 1.0 - ((step - num_warmup_steps) / max(1, total_steps - num_warmup_steps))
                if current_factor < 0:
                    current_factor = 0
                    if not warned_step:
                        logging.warning(f'LR Scheduler: step={step}, total_steps={total_steps}')
                        warned_step = True
            return current_factor

        self.optimizer = torch.optim.Adam(parameters, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lr_lambda=_get_lr)

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def get_lr(self, index: int = 0):
        return self.scheduler.get_last_lr()[index]

    def get_all_lrs(self) -> List[float]:
        return self.scheduler.get_last_lr()
