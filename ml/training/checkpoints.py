import os
import torch
import torch.nn as nn
from typing import NamedTuple, Union

from ml.evaluation import evaluate


Checkpoint = NamedTuple("Checkpoint", [
    ("model", str),
    ("optim", Union[str, None]),
    ("scheduler", Union[str, None]),
    ("stats", Union[str, None]),
])


def save_checkpoint(name: str,
                    model: Union[nn.Module, dict],
                    optim: Union[dict, torch.optim.Optimizer, None] = None,
                    scheduler: Union[dict, torch.optim.lr_scheduler.LRScheduler] = None,
                    stats: Union[list[evaluate], None] = None):
    # Make directories
    if not os.path.exists(name):
        os.makedirs(name)

    # Make names
    model_name = os.path.join(name, "model.pt")
    stats_name = os.path.join(name, "stats.pt")
    optim_name = os.path.join(name, "optim.pt")
    scheduler_name = os.path.join(name, "scheduler.pt")

    files = [stats_name, model_name, optim_name, scheduler_name]

    # Remove previous files
    for name in files:
        if os.path.exists(name):
            os.remove(name)

    # Save everything
    torch.save(model if isinstance(model, dict) else model.state_dict(), model_name)
    if optim:
        torch.save(optim if isinstance(optim, dict) else optim.state_dict(), optim_name)
    if scheduler:
        torch.save(scheduler if isinstance(scheduler, dict) else scheduler.state_dict(), scheduler_name)
    if stats:
        torch.save(stats, stats_name)

    result = Checkpoint(model=model_name,
                        optim=optim_name if optim else None,
                        scheduler=scheduler_name if scheduler else None,
                        stats=stats_name if stats else None)

    return result


def load_checkpoint(name: Union[str, Checkpoint],
                    model: Union[nn.Module, None] = None,
                    optim: Union[torch.optim.Optimizer, None] = None,
                    scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None] = None,
                    compile=True):
    # Make names
    if isinstance(name, Checkpoint):
        model_name = name.model
        stats_name = name.stats
        optim_name = name.optim
        scheduler_name = name.scheduler
    else:
        model_name = os.path.join(name, "model.pt")
        stats_name = os.path.join(name, "stats.pt")
        optim_name = os.path.join(name, "optim.pt")
        scheduler_name = os.path.join(name, "scheduler.pt")

    # Load the model
    if model is not None:
        model.load_state_dict(torch.load(model_name))

    # Load the optim
    if optim is not None:
        optim_w = torch.load(optim_name)
        if not compile:
            optim_w = {k.replace("_orig_mod.", ""): v for k, v in optim_w.items()}

        optim.load_state_dict(optim_w)

    # Load the scheduler
    if scheduler is not None:
        scheduler.load_state_dict(torch.load(scheduler_name))

    # Load the stats
    stats = torch.load(stats_name) if stats_name is not None and os.path.exists(stats_name) else None

    return stats
