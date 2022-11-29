import math
import warnings
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import pytorch_lightning as pl
import torch
from jinja2 import Template
from loguru import logger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast

import wandb
from fz_openqa.callbacks import templates
from fz_openqa.modeling import Model
from fz_openqa.modeling.datastruct import METRIC_PREFIX
from fz_openqa.modeling.modules import ReaderRetriever
from fz_openqa.utils.exceptions import catch_exception_as_warning


class LogRetrievedDocumentsCallback(Callback):
    """
    Log the retrieved documents.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        *,
        log_dir: Path = None,
        n_samples: int = 20,
        top_k: int = 5,
    ):
        super(LogRetrievedDocumentsCallback, self).__init__()
        self.tokenizer = tokenizer
        if log_dir is None:
            self.log_dir = Path()
        self.log_dir = Path(log_dir)
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.n_samples = n_samples
        self.top_k = top_k
        self.data = []

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.data = []

    @catch_exception_as_warning
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        required_keys = [
            "document.input_ids",
            "question.input_ids",
        ]
        if any(c not in batch for c in required_keys):
            warnings.warn(
                "Skipping LogRetrievedDocumentsCallback because the required columns "
                f"{required_keys} are not in the outputs. Found {list(outputs.keys())}"
            )
            return

        if len(self.data) >= self.n_samples:
            return

        # fetch data
        keys = [
            "question.input_ids",
            "document.input_ids",
            f"{METRIC_PREFIX}retriever.logits",
            "document.proposal_score",
        ]
        shape = batch["question.input_ids"].shape[:-1]
        batch = {k: v.detach().cpu() for k, v in batch.items() if k in keys}

        # flatten
        batch = {k: v.reshape(-1, *v.shape[len(shape) :]) for k, v in batch.items()}

        # gather the documents
        for i in range(math.prod(shape)):
            if len(self.data) >= self.n_samples:
                break
            self.data += [{k: v[i] for k, v in batch.items()}]

    @catch_exception_as_warning
    @rank_zero_only
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        retriever_model = self._get_retriever(pl_module)

        with open(Path(templates.__file__).parent / "retrieved_documents.html", "r") as f:
            html_template = Template(f.read())
        html = html_template.render(
            {
                "info": type(retriever_model).__name__,
                "instances": [self._format(row) for row in self.data],
            }
        )

        # save the HTML file
        output = self.log_dir / "retrieved_documents.html"
        with open(output, "w") as f:
            f.write(html)

        try:
            name = "retrieved_documents/html"
            wandb.log({name: wandb.Html(html, inject=False)}, commit=False)
        except wandb.errors.Error as e:
            logger.warning(e)

    def _format(self, row: Dict) -> Dict:
        q_str = self.tokenizer.decode(row["question.input_ids"], skip_special_tokens=True)
        docs = []
        document_input_ids = row["document.input_ids"]
        doc_scores = row.get(f"{METRIC_PREFIX}retriever.logits", None)
        proposal_scores = row.get("document.proposal_score", None)
        for i in range(document_input_ids.shape[0]):
            doc_str = self.tokenizer.decode(document_input_ids[i], skip_special_tokens=True)

            docs.append(
                {
                    "text": doc_str,
                    "retriever_score": doc_scores[i].item() if doc_scores is not None else None,
                    "proposal_score": proposal_scores[i].item()
                    if proposal_scores is not None
                    else None,
                }
            )

        return {"question": q_str, "documents": docs}

    def _get_retriever(self, pl_module: "pl.LightningModule") -> Optional[PreTrainedModel]:

        if not isinstance(pl_module, Model):
            return None

        if not isinstance(pl_module.module, ReaderRetriever):
            return None

        if hasattr(pl_module.module, "retriever"):
            return pl_module.module.retriever

        return None
