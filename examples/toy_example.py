import logging
import os
import sys
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import rich
import torch
import torchvision.transforms
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from rich.status import Status
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from fz_openqa.modeling.parameters import Parameters
from fz_openqa.utils import maybe_instantiate

sys.path.append(str(Path(__file__).parent.parent))

from fz_openqa import configs
from fz_openqa.modeling.gradients import InBatchGradients, ContrastiveGradients
from fz_openqa.modeling.gradients import ReinforceGradients
from fz_openqa.modeling.modules import OptionRetriever
from fz_openqa.toys.dataset import generate_toy_datasets
from fz_openqa.toys.model import ToyOptionRetriever
from fz_openqa.toys.running_stats import RunningStats
from fz_openqa.toys.sampler import ToySampler, SplitWrapper
from fz_openqa.utils.pretty import pprint_batch

logger = logging.getLogger(__name__)
import wandb

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)


def norm(x):
    return np.linalg.norm(x.reshape(-1), 2)


def finish_wandb():
    try:
        wandb.finish()
    except Exception:
        pass


def format_diagnostics(diagnostics: Dict[str, torch.Tensor], split="train"):
    def format(x):
        if isinstance(x, torch.Tensor):
            return x.mean()
        else:
            return x

    return {f"{split}/{k}": format(v) for k, v in diagnostics.items() if not str(k).startswith("_")}


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="toy_config.yaml",
)
def run(config: DictConfig) -> None:
    # torch.multiprocessing.set_start_method('spawn')
    # initialization
    if config.seed is not None:
        seed_everything(config.seed)
    if config.wandb:
        finish_wandb()
        logger = WandbLogger(
            entity="findzebra",
            project="findzebra-qa-toys",
            group="mnist-toy",
            name=config.get("wandb_name", None),
        )
        config_ = deepcopy(OmegaConf.to_object(config))
        config_.pop("sys")
        logger.log_hyperparams(config_)
    else:
        logger = None
    datasets, knowledge = load_datasets(config)
    model = ToyOptionRetriever(
        hidden=config.hidden_size,
        output_size=config.output_size,
        max_chunksize=config.max_chunksize,
        temperature=config.temperature,
        share_backbone=config.share_backbone,
        n_classes=len(config.labels),
    )
    model.to(config.device)
    reinforce_args = {
        "use_baseline": config.use_baseline,
        "baseline_dtype": torch.float16,
        "max_baseline_samples": config.max_baseline_samples,
    }
    estimator_cls, estimator_args = {
        "in_batch": (InBatchGradients, {}),
        "reinforce_a": (ReinforceGradients, {"expr": "A", **reinforce_args}),
        "reinforce_a2": (ReinforceGradients, {"expr": "A2", **reinforce_args}),
        "reinforce_b": (ReinforceGradients, {"expr": "B", **reinforce_args}),
        "reinforce_b_zero": (ReinforceGradients, {"expr": "B-zero", **reinforce_args}),
        "reinforce_c": (ReinforceGradients, {"expr": "C", **reinforce_args}),
        "contrastive": (ContrastiveGradients, {}),
    }[config.estimator]
    estimator = estimator_cls(**estimator_args)
    sampler = ToySampler(
        data={k: v["data"] for k, v in datasets.items()},
        targets={k: v["targets"] for k, v in datasets.items()},
        knowledge=knowledge,
        s_range=None,
        batch_size=config.batch_size,
        n_samples=config.n_samples,
        sampler=config.sampler,
        n_classes=len(config.labels),
        chunksize=config.max_chunksize,
        sample_device=config.device,
    )

    # parameters
    if "parameters" not in config.keys():
        parameters = Parameters()
    elif isinstance(config.parameters, (dict, DictConfig)):
        if "_target_" in config.parameters.keys():
            parameters = maybe_instantiate(config.parameters)
        else:
            parameters = Parameters(**config.parameters)
    else:
        parameters = config.parameters
    assert isinstance(parameters, Parameters)

    # training
    if config.share_backbone:
        param_groups = model.parameters()
    else:
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters() if "reader" in n],
                "lr": config.reader_lr,
            },
            {
                "params": [p for n, p in model.named_parameters() if "retriever" in n],
                "lr": config.retriever_lr,
            },
        ]
    optimizer = torch.optim.Adam(param_groups, lr=config.lr)
    num_epochs = config.num_epochs
    global_step = 0
    gradients = defaultdict(RunningStats)
    for epoch in range(num_epochs):
        train_accuracy = Accuracy().to(config.device)
        if epoch % config.update_freq == 0:
            rich.print(f"> epoch={epoch}, indexing dataset..")
            with Status("Indexing dataset.."):
                model.eval()
                sampler.index(model)
                if logger is not None:
                    logger.log_metrics({"dataset_update/step": epoch}, step=global_step)
        evaluate(
            model,
            estimator,
            sampler,
            config.device,
            logger=logger,
            global_step=global_step,
            epoch=epoch,
            loader_kwargs={
                "batch_size": config.batch_size,
                "num_workers": config.num_workers,
                "pin_memory": config.pin_memory,
            },
            labels=config.labels,
        )

        train_loader = DataLoader(
            SplitWrapper(sampler, split="train"),
            shuffle=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        for i, batch in enumerate(train_loader):
            model.train()
            global_step += 1
            batch = {k: v.to(config.device) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(q=batch["question"], d=batch["evidence"])
            diagnostics = estimator(
                reader_score=output["reader_score"],
                retriever_score=output["retriever_score"],
                targets=batch["target"],
                proposal_score=batch["proposal_score"],
                proposal_log_weight=batch["proposal_log_weight"],
                **parameters(),
            )

            loss = diagnostics["loss"]
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            parameters.step()

            for k, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    gradients[k].push(p.grad.data.clone(), per_dim=True)

            optimizer.zero_grad()

            # update accuracy
            preds = diagnostics["_reader_logits_"].argmax(-1)
            train_accuracy.update(preds, batch["target"])

            if logger is not None:
                pprint_batch(format_diagnostics(diagnostics, split="train"), "diags", silent=True)
                logger.log_metrics(format_diagnostics(diagnostics, split="train"), step=global_step)
                if parameters is not None:
                    logger.log_metrics(
                        format_diagnostics(parameters(), split="parameters"), step=global_step
                    )

            if global_step % 10 == 0:
                train_acc = train_accuracy.compute()
                train_accuracy.reset()
                rich.print(
                    f"> [cyan]train[/cyan] | epoch={epoch}, "
                    f"step={global_step}, "
                    f"acc={train_acc:.3f}"
                )

                if logger is not None:
                    logger.log_metrics(
                        {"train/accuracy": train_acc, "train/loss": loss.mean().detach()},
                        step=global_step,
                    )

            if global_step % 50 == 0:
                evaluate(
                    model,
                    estimator,
                    sampler,
                    config.device,
                    logger=logger,
                    global_step=global_step,
                    epoch=epoch,
                    loader_kwargs={
                        "batch_size": config.batch_size,
                        "num_workers": config.num_workers,
                        "pin_memory": config.pin_memory,
                    },
                    labels=config.labels,
                )

                # print gradients and reset
                if logger is None:
                    for k, g in gradients.items():
                        rich.print(f"{k}: norm={norm(g.mean) :.3E}")
                else:

                    # reduce gradients
                    reduced_gradients = {}
                    if any("reader" in k for k in gradients.keys()):
                        reduced_gradients["reader"] = reduce(
                            RunningStats.cat,
                            (g.flatten() for k, g in gradients.items() if "reader" in k),
                        )
                    if any("retriever" in k for k in gradients.keys()):
                        reduced_gradients["retriever"] = reduce(
                            RunningStats.cat,
                            (g.flatten() for k, g in gradients.items() if "retriever" in k),
                        )

                    # log the gradients
                    logger.log_metrics(
                        format_diagnostics(
                            {k: norm(g.mean) for k, g in reduced_gradients.items()},
                            split="gradients/norm",
                        ),
                        step=global_step,
                    )
                    logger.log_metrics(
                        format_diagnostics(
                            {k: np.mean(g.std) for k, g in reduced_gradients.items()},
                            split="gradients/mean-std",
                        ),
                        step=global_step,
                    )
                    logger.log_metrics(
                        format_diagnostics(
                            {
                                k: np.mean(np.absolute(g.mean[g.std > 0]) / g.std[g.std > 0])
                                for k, g in reduced_gradients.items()
                            },
                            split="gradients/mean-snr",
                        ),
                        step=global_step,
                    )

                gradients = defaultdict(RunningStats)

    if logger is not None:
        logger.finalize("completed")


def load_datasets(config):
    datasets = generate_toy_datasets(
        labels=config.labels,
        root=config.sys.cache_dir,
        download=True,
        noise_level=config.noise_level,
    )
    for key, v in datasets.items():
        if isinstance(v, torch.Tensor):
            rich.print(f"{key}: {v.shape}")
        else:
            for subkey, w in v.items():
                rich.print(f"{key}/{subkey}: {w.shape}")

    knowledge = datasets.pop("knowledge")
    return datasets, knowledge


@torch.no_grad()
def evaluate(
    model,
    estimator,
    sampler,
    device,
    logger=None,
    global_step=None,
    epoch=None,
    loader_kwargs=None,
    labels=None,
):
    model.eval()
    test_accuracy = Accuracy().to(device)
    if loader_kwargs is None:
        loader_kwargs = {}
    loader = DataLoader(
        SplitWrapper(sampler, split="test"),
        shuffle=True,
        # worker_init_fn=seed_everything,
        **loader_kwargs,
    )

    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(q=batch["question"], d=batch["evidence"])
        diagnostics = estimator(
            reader_score=output["reader_score"],
            retriever_score=output["retriever_score"],
            targets=batch["target"],
            proposal_score=batch["proposal_score"],
            proposal_log_weight=batch["proposal_log_weight"],
        )

        # update accuracy
        preds = diagnostics["_reader_logits_"].argmax(-1)
        test_accuracy.update(preds, batch["target"])

        # log diagnostics
        if logger is not None:
            logger.log_metrics(format_diagnostics(diagnostics, split="test"))

        # log images
        if logger is not None and i == 0:
            M = 4
            for idx in range(M):
                idx = (idx * (len(batch["question"]) - 1)) // M
                log_predictions(batch, diagnostics, output, idx=idx, labels=labels)

            # log embeddings
            embs = model.get_embedding_as_image("reader")
            embs = torchvision.utils.make_grid(embs, nrow=embs.size(0)).cpu().numpy().mean(0)
            wandb.log({"embeddings/reader": wandb.Image(embs)})

            embs = model.get_embedding_as_image("retriever")
            embs = torchvision.utils.make_grid(embs, nrow=embs.size(0)).cpu().numpy().mean(0)
            wandb.log({"embeddings/retriever": wandb.Image(embs)})

    test_acc = test_accuracy.compute()
    rich.print(
        f"> [magenta]test[/magenta] | epoch={epoch}, " f"step={global_step}, " f"acc={test_acc:.3f}"
    )
    if logger is not None:
        logger.log_metrics({"test/accuracy": test_acc}, step=global_step)


def log_predictions(batch, diagnostics, output, idx: int = 0, labels=None):
    q = batch["question"][idx].cpu().numpy().mean(0)
    d = batch["evidence"][idx]
    m, n, *dims = d.shape
    pids = output["retriever_score"][idx].argsort(dim=-1, descending=True)
    pids = pids.view(m, n, *(1 for _ in dims)).expand_as(d)
    d = d.gather(1, index=pids)
    # d = d.transpose(1, 0).contiguous()
    d = d.view(-1, *dims)
    d = torchvision.utils.make_grid(d, nrow=n).cpu().numpy().mean(0)
    # probs
    probs = diagnostics["_reader_logits_"][idx].softmax(-1).cpu().numpy()
    probs = [f"{p:.2f}" for p in probs]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, n]})
    ax[0].imshow(q, cmap="gray")
    ax[1].imshow(d, cmap="gray")
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[1].set_title(f"probs={probs}, labels={labels}")
    fig.tight_layout()
    wandb.log({f"predictions/{idx}": fig})
    plt.close(fig)


if __name__ == "__main__":
    run()
