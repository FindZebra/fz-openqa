from enum import Enum
from typing import Any
from typing import Optional

import rich
from datasets import Split
from torch import Tensor

from ...utils.pretty import pprint_batch
from .base import Module
from .utils.concatenate import concat_questions_and_documents
from .utils.concatenate import stack_questions_and_documents
from fz_openqa.utils.datastruct import Batch


class ConcatStrategy(Enum):
    CONCAT = "concat"
    STACK = "stack"


class MedQaReader(Module):
    # name of the features required for a forward pass
    _required_feature_names = [
        "answer.target",
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
    ]

    # named of the features required for evaluation
    _required_eval_feature_names = ["answer.target", "document.match_score"]

    # prefix for the logged metrics
    task_id: Optional[str] = "reader"

    # metrics to display
    pbar_metrics = [
        "train/reader/Accuracy",
        "validation/reader/Accuracy",
        "train/reader/relevance-Accuracy",
        "validation/reader/relevance-Accuracy",
    ]

    _required_heads = ["option", "evidence", "relevance"]

    def __init__(self, *args, concat_strategy: ConcatStrategy = ConcatStrategy.CONCAT, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_strategy = ConcatStrategy(concat_strategy)

    def _init_metrics(self, prefix: str = ""):
        """Initialize a Metric for each split=train/validation/test
        fir both the answering model and the selection model"""
        self.answer_metrics = self._get_base_metrics(prefix=prefix)

    def _forward(self, batch: Batch, targets: Optional[Tensor] = None, **kwargs) -> Batch:
        # tokenizer = AutoTokenizer.from_pretrained(self.bert.name_or_path, use_fast=True)
        # checks inputs, set parameters and concat the questions with the documents
        # pprint_batch(batch)
        # self.bert.tokenizer

        rich.print(f"[magenta] input question tokens: {batch['question.input_ids'].shape}")
        rich.print(f"[magenta] input document tokens: {batch['document.input_ids'].shape}")

        # concatenate questions and documents such that there is no padding between Q and D
        args = {"pad_token_id": self._pad_token_id, "max_length": self.max_length}
        if self.concat_strategy == ConcatStrategy.CONCAT:
            # returns tokens of shape [bs, n_opts, total_length]
            qd_batch = concat_questions_and_documents(batch, **args)
        elif self.concat_strategy == ConcatStrategy.STACK:
            # returns tokens of shape [bs * n_docs, n_opts, total_length]
            qd_batch = stack_questions_and_documents(batch, **args)
        else:
            raise ValueError(f"Unknown concat strategy: {self.concat_strategy}")

        pprint_batch(qd_batch, "qd_batch")
        # tokenizer = AutoTokenizer.from_pretrained(self.bert.name_or_path, use_fast=True)
        rich.print(f"[cyan] ANS: {self.tokenizer.encode('[ANS]')}")
        rich.print(f"[cyan] QUERY: {self.tokenizer.encode('[QUERY]')}")
        rich.print(f"[cyan] DOC: {self.tokenizer.encode('[DOC]')}")

        rich.print(f"[magenta] padded tokens: {qd_batch['input_ids'].shape}")
        for i in range(min(len(qd_batch["input_ids"]), 16)):
            print(100 * "=")
            rich.print(f"- batch el #{i + 1}")
            tokens = qd_batch["input_ids"][i]
            for j in range(2):
                rich.print(f"- option #{j + 1}")
                print(100 * "-")
                # rich.print(f"[red] {tokens[j].tolist()}")
                decoded = self.tokenizer.decode(tokens[j].tolist())
                rich.print(f"[cyan] {decoded}")
        exit()

        return self.bert(
            qd_batch["input_ids"], qd_batch["attention_mask"], labels=targets, return_dict=True
        )

    def _step(self, batch: Batch, **kwargs: Any) -> Batch:

        answer_targets: Tensor = batch["answer.target"]

        # forward pass through the reader model
        outputs = self._forward(batch, targets=answer_targets, **kwargs)

        return {
            "loss": outputs["loss"],
            "_answer_logits_": outputs["logits"].detach(),
            "_answer_targets_": answer_targets.detach(),
        }

    def _reduce_step_output(self, output: Batch) -> Batch:
        """
        Gather losses from all devides and return
        """

        # average losses
        for k in ["loss"]:
            y = output.get(k, None)
            if y is not None:
                output[k] = y.mean()

        return output

    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""
        answer_logits, answer_targets = (
            output.get(k, None) for k in ("_answer_logits_", "_answer_targets_")
        )
        self.answer_metrics.update(split, answer_logits, answer_targets)

    def reset_metrics(self, split: Optional[Split] = None) -> None:
        """
        Reset the metrics corresponding to `split` if provided, else
        reset all the metrics.
        """
        self.answer_metrics.reset(split)

    def compute_metrics(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        return {
            **self.answer_metrics.compute(split),
        }
