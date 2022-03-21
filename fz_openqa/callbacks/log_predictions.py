from typing import Any
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import spacy
from loguru import logger
from pip._internal import main as pipmain
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from spacy import displacy
from transformers import PreTrainedTokenizerFast

import wandb

CORRECT_LABEL = "✅"
INCORRECT_LABEL = "❌"

SPACY_MODELS = {
    "en_core_sci_sm": "https://s3-us-west-2.amazonaws.com/"
    "ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz",
    "en_core_sci_md": "https://s3-us-west-2.amazonaws.com/"
    "ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz",
    "en_core_sci_lg": "https://s3-us-west-2.amazonaws.com/"
    "ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz",
}


def maybe_download_spacy_model(model_name: str) -> spacy.Language:
    spacy_model_url = SPACY_MODELS.get(model_name, model_name)
    try:
        spacy_model = spacy.load(model_name)
    except OSError:
        if spacy_model_url.startswith("http"):
            pipmain(["install", spacy_model_url])
        else:
            from spacy.cli import download

            download(spacy_model_url)
        spacy_model = spacy.load(model_name)

    return spacy_model


class LogPredictions(Callback):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        *,
        log_dir: str,
        verbose: bool = True,
        n_samples: int = 10,
        spacy_model: Optional[str] = "en_core_sci_md",
    ):
        super(LogPredictions, self).__init__()
        self.tokenizer = tokenizer
        self.log_dir = log_dir
        self.verbose = verbose
        self.n_samples = n_samples
        self.data = []
        if spacy_model is not None:
            self.spacy_model = maybe_download_spacy_model(spacy_model)
        else:
            self.spacy_model = None

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

    def decode(self, input_ids):
        u = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        if self.spacy_model is not None:
            doc = self.spacy_model(u)
            html = displacy.render(doc, style="ent")
        else:
            html = f'<div class="entities" style="line-height: 2.5; direction: ltr">{u}</div>'

        return html + "\n"

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if len(self.data) > self.n_samples:
            self.data = np.random.choice(self.data, self.n_samples, replace=False)

        html = "<h1>Model predictions</h1>\n"
        for k, row in enumerate(self.data):
            html += '<div tag="Question" style="font-size:12px">\n'
            html += f"<h2>Q #{k}</h2>\n"
            probs = row["_reader_logits_"].softmax(-1)
            scores = row["document.retrieval_score"]
            target = row["answer.target"]
            for i, qids in enumerate(row["question.input_ids"]):
                html += "<hr>"
                html += '<div class="option" style="background-color:#eee;">\n'
                label = CORRECT_LABEL if i == target else INCORRECT_LABEL
                html += f"<h3>{label} Opt#{i} (Q#{k}) - prob={probs[i]:.2f}</h3>\n"
                html += self.decode(qids)
                html += '</div">\n'

                # documents
                doc_probs = row["_doc_logits_"][i].softmax(-1)
                js = doc_probs.argsort(-1, descending=True)
                html += '<div class="row" style="background-color:#fff;">\n'
                for _j, j in enumerate(js[:3]):
                    html += '<div tag="document" class="column">\n'
                    html += (
                        f"<h4>Doc #{_j}, retriever_prob={doc_probs[j]:.2f},"
                        f" retrieval_score={scores[i, j]:.2f}</h4>\n"
                    )
                    dids = row["document.input_ids"][i][j]
                    html += self.decode(dids)
                    html += '</div">\n'
                html += "</div>\n"

            html += '</div">\n'

        try:
            name = "predictions"
            wandb.log({name: wandb.Html(html, inject=False)}, commit=False)
        except Exception as e:
            logger.warning(f"Could not log to wandb: {e}")
