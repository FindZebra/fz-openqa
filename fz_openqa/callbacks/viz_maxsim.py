import os
import warnings
from pathlib import Path
from typing import Any
from typing import Dict

import pytorch_lightning as pl
import rich
import seaborn as sns
import torch
from loguru import logger
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities import rank_zero_only
from transformers import PreTrainedTokenizerFast

import wandb
from fz_openqa.modeling.heads import ColbertHead
from fz_openqa.modeling.model import Model
from fz_openqa.modeling.modules import OptionRetriever
from fz_openqa.utils.exceptions import catch_exception_as_warning
from fz_openqa.utils.pretty import pretty_decode


class VizMaxsimCallback(Callback):
    def __init__(
        self,
        max_steps: int = 1,
        max_questions: int = 2,
        max_options: int = 2,
        max_documents: int = 3,
        log_dir: str = None,
        tokenizer: PreTrainedTokenizerFast = None,
        **kwargs,
    ):
        super(VizMaxsimCallback, self).__init__()

        # output directory
        if log_dir is None:
            log_dir = Path(os.getcwd()) / "viz-maxsim"
        self.logdir = Path(log_dir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        # parameters
        self.max_questions = max_questions
        self.max_options = max_options
        self.max_document = max_documents
        self.step_counter = 0
        self.max_steps = max_steps
        self.tokenizer = tokenizer

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.step_counter = 0

    @catch_exception_as_warning
    @rank_zero_only
    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        if self.step_counter >= self.max_steps:
            return
        self.step_counter += 1

        if not self._is_compatible(pl_module):
            return

        # process the batch of data
        batch = self._sample_batch(batch, n=self.max_questions)
        batch = move_data_to_device(batch, device=pl_module.device)
        output = pl_module(batch, head="retriever")
        batch = move_data_to_device(batch, device="cpu")
        output = move_data_to_device(output, device="cpu")

        # make the output dir for that step
        output_dir = self.logdir / f"step-{trainer.global_step}"
        output_dir.mkdir(exist_ok=True, parents=True)
        self._format_and_log_maxsim(batch, output, output_dir)

    def _format_and_log_maxsim(self, batch, output, output_dir):
        with open(output_dir / "outputs.txt", "w") as f:
            H_MAX = 350
            W_MAX = 200
            SIZE_TOKEN_H = 0.12
            SIZE_TOKEN_W = 0.18
            d_input_ids = batch.get("document.input_ids")
            q_input_ids = batch.get("question.input_ids")
            qids = batch.get("question.row_idx")
            dids = batch.get("document.row_idx")
            proposal_score = batch.get("document.proposal_score")
            hq = output.get("_hq_")
            hd = output.get("_hd_")
            qmask = hq.abs().sum(-1) == 0
            dmask = hd.abs().sum(-1) == 0
            scores = torch.einsum("bmqh,bmkdh->bmkqd", hq, hd)
            qmask = qmask[:, :, None, :, None]
            scores = scores.masked_fill(qmask, 0)
            dmask = dmask[:, :, :, None, :]
            scores = scores.masked_fill(dmask, -torch.inf)
            targets = batch.get("answer.target")
            for i in range(min(self.max_questions, len(hq))):
                js = list(range(len(d_input_ids[i])))
                j_target = targets[i].item()
                js = [j_target] + list(set(js) - {j_target})
                for j in js[: self.max_options]:
                    d_input_ids_i = d_input_ids[i][j]
                    q_input_ids_i = q_input_ids[i][j]
                    proposal_score_i = proposal_score[i][j]
                    q_input_ids_i_ = [int(q.item()) for q in q_input_ids_i]

                    if int(self.tokenizer.pad_token_id) in q_input_ids_i_:
                        q_padding_idx = q_input_ids_i_.index(int(self.tokenizer.pad_token_id))
                    else:
                        q_padding_idx = None
                    q_input_ids_i = q_input_ids_i[:q_padding_idx]
                    scores_i = scores[i][j]
                    msg = f" Question ({i + 1}, {j + 1}) "
                    u = pretty_decode(q_input_ids_i, tokenizer=self.tokenizer)
                    f.write(f"{msg}\n{u}\n")

                    for k in range(self.max_document):
                        # hd_ik = hd_i[k]
                        proposal_score_ik = proposal_score_i[k]
                        scores_ik = scores_i[k, :q_padding_idx, :].clone()
                        d_input_ids_ik = d_input_ids_i[k]
                        msg = f" Document {i + 1}-{j + 1}-{k + 1} : score={proposal_score_ik:.2f} "
                        u = pretty_decode(d_input_ids_ik, tokenizer=self.tokenizer, style="white")
                        f.write(f"{msg}\n{u}\n")

                        # min and max values
                        flat_scores = scores_ik.clone().view(-1)
                        flat_scores = flat_scores[(~flat_scores.isnan()) & (~flat_scores.isinf())]
                        MIN_SCORE = flat_scores.min() - 0.2 * (
                            flat_scores.max() - flat_scores.min()
                        )
                        MAX_SCORE = scores.max() + 0.2 * (flat_scores.max() - flat_scores.min())

                        # replace `-inf` with `MIN_SCORE`
                        scores_ik = scores_ik.masked_fill(scores_ik == -torch.inf, MIN_SCORE)

                        # visualize the scores
                        q_tokens = [
                            self.tokenizer.decode(t, skip_special_tokens=False)
                            for t in q_input_ids_i
                        ]
                        d_tokens = [
                            self.tokenizer.decode(t, skip_special_tokens=False)
                            for t in d_input_ids_ik
                        ]

                        # print corresponding `max` scores
                        with open(output_dir / f"max_mapping-{i}-{j}-{k}.txt", "w") as fmp:
                            for qi, q_token_qi in enumerate(q_tokens):
                                qj = scores_ik[qi].argmax(dim=-1)
                                fmp.write(f"{q_token_qi:>30} -> {d_tokens[qj]:<30}\n")

                        # get the scores and reference ids
                        y = scores_ik[:H_MAX, :W_MAX]
                        d_tokens = d_tokens[:W_MAX]
                        q_tokens = q_tokens[:H_MAX]
                        uqid = qids[i]
                        udid = dids[i, j, k]

                        # highlight the max score

                        yms = torch.zeros_like(y[:, 0])
                        for u in range(y.shape[0]):
                            ym_u = y[u].max().item()
                            yms[u] = ym_u
                            for v in range(y.shape[1]):
                                if y[u, v] >= ym_u:
                                    y[u, v] = MAX_SCORE

                        # Append the column of query token scores: max_j s_ij
                        d_tokens = ["[MAX]", ""] + d_tokens
                        yms = yms.unsqueeze(1)
                        y = torch.cat([yms, MIN_SCORE * torch.ones_like(yms), y], dim=1)

                        # heatmap
                        fig = plt.figure(
                            figsize=(int(SIZE_TOKEN_W * y.shape[1]), int(SIZE_TOKEN_H * y.shape[0]))
                        )
                        sns.heatmap(
                            y,
                            xticklabels=d_tokens,
                            yticklabels=q_tokens,
                            vmin=MIN_SCORE,
                            vmax=MAX_SCORE,
                        )
                        plt.savefig(output_dir / f"heatmap-{uqid}-{j}-{udid}.png")
                        try:
                            wandb_img = wandb.Image(fig)
                            wandb.log({f"maxsim/heatmap-{uqid}-{j}-{udid}.png": wandb_img})
                        except wandb.errors.Error as exc:
                            logger.warning(exc)
                        plt.close()

    def _sample_batch(self, batch: Dict, n: int = None) -> Dict:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[:n]

        return batch

    def _is_compatible(self, pl_module) -> bool:
        if not isinstance(pl_module, Model):
            warnings.warn(
                f"{type(self).__name__} requires model of type Model. "
                f"Found {type(pl_module).__name__}"
            )
            return False

        elif not isinstance(pl_module.module, OptionRetriever):
            warnings.warn(
                f"{type(self).__name__} requires module of type OptionRetriever. "
                f"Found {type(pl_module.module).__name__}"
            )
            return False

        elif not isinstance(pl_module.module.retriever_head, ColbertHead):
            warnings.warn(
                f"{type(self).__name__} requires head of type ColbertHead. "
                f"Found {type(pl_module.moduleretriever_head).__name__}"
            )
            return False
        else:
            return True
