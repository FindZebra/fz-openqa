from typing import Any
from typing import Optional

import pytorch_lightning as pl
import spacy
from loguru import logger
from pip._internal import main as pipmain
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from spacy import displacy
from transformers import PreTrainedTokenizerFast

import wandb
from fz_openqa.utils.exceptions import catch_exception_as_warning

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
        spacy_model: Optional[str] = None,
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

        if "_reader_logits_" not in outputs:
            return

        if len(self.data) >= self.n_samples:
            return

        preds = outputs["_reader_logits_"].argmax(dim=-1).cpu().detach()
        targets = batch["answer.target"].cpu().detach()

        batch_keys = [
            "question.input_ids",
            "document.input_ids",
            "answer.target",
            "document.proposal_score",
        ]
        preds_keys = [
            "_reader_logits_",
            "_retriever_scores_",
            "_reader_scores_",
        ]
        out = {k: v for k, v in batch.items() if k in batch_keys}
        out.update({k: v for k, v in outputs.items() if k in preds_keys})

        # handle the document_ids in the case where documents are shared across the batch
        if "_document.inv_ids_" in outputs:
            uids = outputs["_document.inv_ids_"]
            doc_input_ids = out["document.input_ids"]
            bs, n_opts, n_docs, seq_length = doc_input_ids.shape
            doc_input_ids = doc_input_ids.view(-1, seq_length)
            doc_input_ids = doc_input_ids[uids]
            doc_input_ids = doc_input_ids.view(1, 1, len(uids), seq_length)
            doc_input_ids = doc_input_ids.expand(bs, n_opts, len(uids), seq_length)
            out["document.input_ids"] = doc_input_ids

        # select the rows where the prediction was correct
        mask = preds == targets
        out = {k: v[mask] for k, v in out.items()}

        for i in range(len(out["question.input_ids"])):
            if len(self.data) >= self.n_samples:
                break
            self.data += [{k: v[i] for k, v in out.items()}]

    def decode(self, input_ids):
        u = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        if self.spacy_model is not None:
            doc = self.spacy_model(u)
            html = displacy.render(doc, style="ent")
        else:
            html = f'<div class="entities" style="line-height: 2.5; direction: ltr">{u}</div>'

        return html + "\n"

    @catch_exception_as_warning
    @rank_zero_only
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        html = "<h1>Model predictions</h1>\n"
        for k, row in enumerate(self.data):
            html += '<div tag="Question" style="font-size:12px">\n'
            html += f"<h2>Q #{k}</h2>\n"
            probs = row["_reader_logits_"].softmax(-1)
            proposal_scores = row["document.proposal_score"]
            target = row["answer.target"]
            doc_input_ids = row["document.input_ids"]

            # display each question
            for i, qids in enumerate(row["question.input_ids"]):
                is_correct = i == target
                html += "<hr>"
                html += '<div class="option" style="background-color:#eee;">\n'
                label = CORRECT_LABEL if is_correct else INCORRECT_LABEL
                html += f"<h3>{label} Opt#{i} (Q#{k}) - prob={probs[i]:.2f}</h3>\n"
                html += self.decode(qids)
                html += '</div">\n'

                # documents
                doc_scores = row["_retriever_scores_"][i]
                doc_probs = doc_scores.softmax(-1)
                reader_scores = row["_reader_scores_"][i]
                sort_score = doc_scores + reader_scores
                js = sort_score.argsort(-1, descending=True)
                html += '<div class="row" style="background-color:#fff;">\n'
                for _j, j in enumerate(js[:3]):
                    html += '<div tag="document" class="column">\n'
                    try:
                        # cannot be retrieved when setting `share_documents_across_batch`
                        proposal_score_ = f"{proposal_scores[i, j]:.2f}"
                    except IndexError:
                        proposal_score_ = "--"

                    try:
                        # cannot be retrieved when using `ContrastiveGradients`
                        reader_score_ = f"{reader_scores[j]:.2f}"
                    except IndexError:
                        reader_score_ = "--"

                    html += (
                        f"<h4>Doc #{_j}, "
                        f"retriever_prob={doc_probs[j]:.2f}, "
                        f"proposal_score={proposal_score_}, "
                        f"reader_score={reader_score_}, "
                        f"</h4>\n"
                    )
                    dids = doc_input_ids[i][j]
                    html += self.decode(dids)
                    html += '</div">\n'
                html += "</div>\n"

            html += '</div">\n'

        try:
            name = "predictions/htm"
            wandb.log({name: wandb.Html(html, inject=False)}, commit=False)
        except wandb.errors.Error as e:
            logger.warning(e)
