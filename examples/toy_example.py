import logging
from copy import deepcopy
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
from torchmetrics import Accuracy

from fz_openqa import configs
from fz_openqa.modeling.gradients import InBatchGradients
from fz_openqa.modeling.gradients import ReinforceGradients
from fz_openqa.modeling.modules import OptionRetriever
from fz_openqa.toys.dataset import generate_toy_datasets
from fz_openqa.toys.model import ToyOptionRetriever
from fz_openqa.toys.sampler import ToySampler
from fz_openqa.utils.pretty import pprint_batch

logger = logging.getLogger(__name__)
import wandb


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
    # initialization
    if config.seed is not None:
        seed_everything(config.seed)
    if config.wandb:
        finish_wandb()
        logger = WandbLogger(entity="findzebra", project="findzebra-qa-toys", group="mnist-toy")
        config_ = deepcopy(OmegaConf.to_object(config))
        config_.pop("sys")
        logger.log_hyperparams(config_)
    else:
        logger = None
    datasets, knowledge = load_datasets(config)
    model = ToyOptionRetriever(
        hidden=config.hidden_size, max_chunksize=10_000, share_backbone=config.share_backbone
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
        "reinforce_b": (ReinforceGradients, {"expr": "B", **reinforce_args}),
        "reinforce_b_zero": (ReinforceGradients, {"expr": "B-zero", **reinforce_args}),
        "reinforce_c": (ReinforceGradients, {"expr": "C", **reinforce_args}),
    }[config.estimator]
    estimator = estimator_cls(**estimator_args)
    sampler = ToySampler(
        data={k: v["data"].to(config.device) for k, v in datasets.items()},
        targets={k: v["targets"].to(config.device) for k, v in datasets.items()},
        knowledge=knowledge.to(config.device),
        s_range=100,
        batch_size=config.batch_size,
        n_samples=config.n_samples,
    )

    # training
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    num_epochs = config.num_epochs
    global_step = 0
    for epoch in range(num_epochs):
        train_accuracy = Accuracy().to(config.device)
        with Status("Indexing dataset.."):
            sampler.index(model if epoch > 0 else None)
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
        )
        for i, batch in enumerate(sampler.iter_split("train")):
            global_step += 1
            optimizer.zero_grad()
            output = model.compute_score(q=batch["question"], d=batch["evidence"])
            diagnostics = estimator(
                reader_score=output["reader_score"],
                retriever_score=output["retriever_score"],
                targets=batch["target"],
                retrieval_score=batch["retrieval_score"],
                retrieval_log_weight=batch["retrieval_log_weight"],
            )

            loss = diagnostics["loss"].mean()
            loss.backward()
            optimizer.step()

            # retriever diagnostics
            OptionRetriever._retriever_diagnostics(
                retriever_score=output["retriever_score"],
                retrieval_scores=batch["retrieval_score"],
                retrieval_rank=batch["retrieval_score"].argsort(dim=-1, descending=True),
                output=diagnostics,
            )

            # update accuracy
            preds = diagnostics["_reader_logits_"].argmax(-1)
            train_accuracy.update(preds, batch["target"])

            if logger is not None:
                pprint_batch(format_diagnostics(diagnostics, split="train"), "diags", silent=True)
                logger.log_metrics(format_diagnostics(diagnostics, split="train"), step=global_step)

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
                        {"train/accuracy": train_acc},
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
                )

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
def evaluate(model, estimator, sampler, device, logger=None, global_step=None, epoch=None):
    test_accuracy = Accuracy().to(device)
    for i, batch in enumerate(sampler.iter_split("test")):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model.compute_score(q=batch["question"], d=batch["evidence"])
        diagnostics = estimator(
            reader_score=output["reader_score"],
            retriever_score=output["retriever_score"],
            targets=batch["target"],
            retrieval_score=batch["retrieval_score"],
            retrieval_log_weight=batch["retrieval_log_weight"],
        )

        # update accuracy
        preds = diagnostics["_reader_logits_"].argmax(-1)
        test_accuracy.update(preds, batch["target"])

        # log diagnostics
        if logger is not None:
            logger.log_metrics(format_diagnostics(diagnostics, split="test"))

        # log images
        if logger is not None and i == 0:
            q = batch["question"][0].cpu().numpy().mean(0)
            d = batch["evidence"][0]
            m, n, *dims = d.shape
            idx = output["retriever_score"][0].argsort(dim=-1, descending=True)
            idx = idx.view(m, n, *(1 for _ in dims)).expand_as(d)
            d = d.gather(1, index=idx)

            # d = d.transpose(1, 0).contiguous()
            d = d.view(-1, *dims)
            d = torchvision.utils.make_grid(d, nrow=n).cpu().numpy().mean(0)

            # probs
            probs = diagnostics["_reader_logits_"][0].softmax(-1).cpu().numpy()
            probs = [f"{p:.2f}" for p in probs]

            fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, n]})
            ax[0].imshow(q, cmap="gray")
            ax[1].imshow(d, cmap="gray")
            ax[0].set_xticks([])
            ax[1].set_xticks([])
            ax[0].set_yticks([])
            ax[1].set_yticks([])
            ax[1].set_title(f"probs={probs}")
            fig.tight_layout()
            wandb.log({"predictions": fig})
            plt.close(fig)

    test_acc = test_accuracy.compute()
    rich.print(
        f"> [magenta]test[/magenta] | epoch={epoch}, " f"step={global_step}, " f"acc={test_acc:.3f}"
    )
    if logger is not None:
        logger.log_metrics({"test/accuracy": test_acc}, step=global_step)


if __name__ == "__main__":
    run()
