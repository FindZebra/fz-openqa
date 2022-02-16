import logging
import shutil
from io import StringIO
from typing import Any
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import rich
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from rich.console import Console
from transformers import PreTrainedTokenizerFast

from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch
from fz_openqa.utils.pretty import pretty_decode

logger = logging.getLogger(__name__)


class LogPredictions(Callback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        *,
        log_dir: str,
        verbose: bool = True,
        n_samples: int = 10,
    ):
        super(LogPredictions, self).__init__()
        self.tokenizer = tokenizer
        self.log_dir = log_dir
        self.verbose = verbose
        self.n_samples = n_samples
        self.data = []

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.data = []

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        preds = outputs["_reader_logits_"].argmax(dim=-1).cpu().detach()
        targets = batch["answer.target"].cpu().detach()
        mask = preds == targets

        batch_keys = [
            "question.input_ids",
            "document.input_ids",
            "answer.target",
            "document.retrieval_score",
        ]
        preds_keys = ["_reader_logits_", "_doc_logits_"]
        out = {k: v[mask] for k, v in batch.items() if k in batch_keys}
        out.update({k: v[mask] for k, v in outputs.items() if k in preds_keys})

        for i in range(len(out["question.input_ids"])):
            self.data += [{k: v[i] for k, v in out.items()}]

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        if len(self.data) > self.n_samples:
            self.data = np.random.choice(self.data, self.n_samples, replace=False)

        console = Console(record=True, file=None if self.verbose else StringIO())
        repr = ""
        for row in self.data:
            probs = row["_reader_logits_"].softmax(-1)
            scores = row["document.retrieval_score"]
            repr += get_separator("=")
            target = row["answer.target"]
            for i, qids in enumerate(row["question.input_ids"]):
                u = pretty_decode(
                    qids, tokenizer=self.tokenizer, style="green" if i == target else "cyan"
                )
                repr += get_separator("-")
                repr += f"(i={i}, p={probs[i]:.2f}) {u}\n"
            for i, qids in enumerate(row["question.input_ids"]):
                doc_probs = row["_doc_logits_"][i].softmax(-1)
                js = doc_probs.argsort(-1, descending=True)
                for j in js[:2]:
                    dids = row["document.input_ids"][i][j]
                    u = pretty_decode(dids, tokenizer=self.tokenizer)
                    repr += get_separator(".")
                    repr += f"(i={i}, j={j}, p={doc_probs[j]:.2f}, s={scores[i, j]:.2f}) {u}\n"

        console.print(repr)
        html = console.export_html()

        try:
            name = "predictions"
            wandb.log({name: wandb.Html(html, inject=False)}, commit=False)
        except Exception as e:
            logger.warning(f"Could not log to wandb: {e}")
