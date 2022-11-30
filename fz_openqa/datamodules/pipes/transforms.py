import warnings
from typing import Dict
from typing import List
from typing import Optional

import rich
import torch
import torch.nn.functional as F
from datasets import Split
from transformers import PreTrainedTokenizerFast
from warp_pipes import Batch
from warp_pipes import Flatten
from warp_pipes import Nest
from warp_pipes import Pipe
from warp_pipes import pprint_batch
from warp_pipes.core.condition import In
from warp_pipes.support.functional import iter_batch_egs
from warp_pipes.support.shapes import infer_shape

from fz_openqa.datamodules.builders.utils.format_row import infer_scenario
from fz_openqa.datamodules.utils.datastruct import Scenario
from fz_openqa.modeling.templates import LanguageModellingTemplateGenerator


class Transform(Pipe):
    """
    A pipe to transform a batch of examples to a batch of examples with a different structure.
    """

    def __init__(
        self,
        target_key: str = "answer.target",
        question_id_key: str = "question.id",
        question_loc_key: str = "question.loc",
        keys: List[str] = None,
        input_filter=None,
        splits: Optional[List[Split]] = None,
        **kwargs,
    ):
        keys = keys or ["question.input_ids", "question.attention_mask"]
        self.target_key = target_key
        self.question_id_key = question_id_key
        self.question_loc_key = question_loc_key
        self.keys = keys
        self.splits = splits
        if input_filter is not None:
            raise ValueError("FlattenMcQuestions does not support input_filter")
        input_filter = In(self.keys + [target_key])
        super().__init__(**kwargs, input_filter=input_filter)


class TransformMcQuestions(Transform):
    """
    Transform Multiple-Choice Questions weith input shape [bs, n_opts, *]
    """

    def _gather_features(self, batch: Batch) -> (torch.Size, torch.Tensor, Batch):
        # get the features
        features = {k: v for k, v in batch.items() if k in self.keys}
        eg = [x for x in features.values() if isinstance(x, torch.Tensor)][0]
        n_options = eg.shape[1]
        for k, v in features.items():
            if isinstance(v, torch.Tensor) and not v.shape[1] == n_options:
                raise ValueError(
                    f"Attribute {k} doesn't have the right number of "
                    f"options ({n_options}), found {v.shape[1]} (shape={v.shape})"
                )

        # get the target
        if self.target_key in batch.keys():
            targets = batch[self.target_key]
        else:
            warnings.warn(
                f"No target found in batch for key {self.target_key}, "
                f"setting target to -1. Found keys: {batch.keys()}"
            )
            targets = torch.ones(len(batch), dtype=torch.long) * -1

        return eg.shape[:2], targets, features


class FlattenMcQuestions(TransformMcQuestions):
    """
    Flatten Multiple-Choice Questions: [bs, n_opts, *] -> [bs * n_opts, *].
    The answer target will be binarized in the process.
    """

    @torch.no_grad()
    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:
        if self.splits is not None and split not in self.splits:
            return {}

        pprint_batch(batch, "FlattenMcQuestions::input")

        # get the features
        shape, targets, features = self._gather_features(batch)

        # build binary targets, set all to zero is the target is not provided (targets < 0)
        binary_targets = torch.zeros(*shape, dtype=torch.bool, device=targets.device)
        if (targets >= 0).all():
            binary_targets.scatter_(1, targets.unsqueeze(1), 1)

        # build the `question.id` feature
        question_id = torch.arange(shape[0], dtype=torch.long, device=targets.device)
        question_id = question_id.unsqueeze(1).expand(shape).contiguous()

        # prepare the output
        output = {
            self.target_key: binary_targets,
            self.question_id_key: question_id,
            **features,
        }

        # flatten the features
        output = Flatten(level=1)(output)

        # build the `question.loc` feature
        output[self.question_loc_key] = torch.arange(
            output[self.question_id_key].shape[0], dtype=torch.long, device=targets.device
        )

        pprint_batch(batch, "FlattenMcQuestions::output")

        return output


class OptionDropout(TransformMcQuestions):
    """
    Drop out options during training.
    """

    @torch.no_grad()
    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:

        if self.splits is not None and split not in self.splits:
            return {}

        # get the tensors
        target = batch[self.target_key]
        features = {k: v for k, v in batch.items() if k in self.keys}  # todo: refactor
        eg = [x for x in features.values() if isinstance(x, torch.Tensor)][0]
        n_options = eg.shape[1]
        for k, v in features.items():
            if isinstance(v, torch.Tensor) and not v.shape[1] == n_options:
                raise ValueError(
                    f"Attribute {k} doesn't have the right number of "
                    f"options ({n_options}), found {v.shape[1]} (shape={v.shape})"
                )

        # define sampling probs
        logits = torch.ones(eg.shape[0], n_options, dtype=torch.float32, device=eg.device)
        logits.scatter_(dim=1, index=target.unsqueeze(1), value=-float("inf"))

        # sample labels #2:
        rdn_idx = F.gumbel_softmax(logits, hard=False)
        rdn_idx = rdn_idx.argmax(dim=1)

        # gather data
        selected_ids = torch.cat([target[:, None], rdn_idx[:, None]], dim=1)

        # generate the new answer target
        answer_target = torch.zeros_like(rdn_idx)
        answer_target.bernoulli_(0.5)

        # shuffle selected ids
        mask = torch.cat([answer_target[:, None], 1 - answer_target[:, None]], dim=1)
        selected_ids = selected_ids.gather(dim=1, index=mask)

        # for i in range(len(target)):
        #     msg = f"# logits={logits[i]}, target={target[i]}, rdn_idx={rdn_idx[i]}"
        #     assert selected_ids[i][answer_target[i]] == target[i]
        #     assert selected_ids[i, 0] != selected_ids[i, 1], msg

        # output
        output = {self.target_key: answer_target}
        for key, x in features.items():
            if isinstance(x, torch.Tensor):
                leaf_shape = x.shape[len(selected_ids.shape) :]
                _index = selected_ids.view(*selected_ids.shape, *(1 for _ in leaf_shape))
                _index = _index.expand(*selected_ids.shape, *leaf_shape)
                output[key] = x.gather(dim=1, index=_index)
            else:
                output[key] = [[y[i] for i in ids] for y, ids in zip(x, selected_ids)]

        # pprint_batch(output, "OptionDropout")
        return output


class OptionBinarized(OptionDropout):
    @torch.no_grad()
    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:

        if self.splits is not None and split not in self.splits:
            return {}

        # get the tensors
        target = batch[self.target_key]
        values = {k: v for k, v in batch.items() if k in self.keys}  # todo: refactor
        eg = [x for x in values.values() if isinstance(x, torch.Tensor)][0]
        n_options = eg.shape[1]
        for k, v in values.items():
            if isinstance(v, torch.Tensor) and not v.shape[1] == n_options:
                raise ValueError(
                    f"Attribute {k} doesn't have the right number of "
                    f"options ({n_options}), found {v.shape[1]} (shape={v.shape})"
                )

        # define sampling probs
        logits = torch.ones(eg.shape[0], n_options, dtype=torch.float32, device=eg.device)

        # sample label
        rdn_idx = F.gumbel_softmax(logits, hard=False)
        rdn_idx = rdn_idx.argmax(dim=1)
        answer_target = rdn_idx == target
        selected_ids = rdn_idx[:, None]

        # output
        output = {self.target_key: answer_target}
        for key, x in values.items():
            if isinstance(x, torch.Tensor):
                leaf_shape = x.shape[len(selected_ids.shape) :]
                _index = selected_ids.view(*selected_ids.shape, *(1 for _ in leaf_shape))
                _index = _index.expand(*selected_ids.shape, *leaf_shape)
                output[key] = x.gather(dim=1, index=_index)
            else:
                output[key] = [[y[i] for i in ids] for y, ids in zip(x, selected_ids)]

        return output


class StripAnswer(Pipe):
    """Remove tokens corresponding to the answer from the question"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        field: str = "question",
        **kwargs,
    ):
        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.field = field
        super().__init__(**kwargs)

    @torch.no_grad()
    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:
        input_ids = batch[f"{self.field}.input_ids"]
        attention_mask = batch[f"{self.field}.attention_mask"]

        # identify the position of the first SEP token
        ids = torch.linspace(0, 0.9999, input_ids.size(-1), device=input_ids.device)
        for _ in range(max(0, input_ids.dim() - ids.dim())):
            ids = ids.unsqueeze(0)
        ids = ids + 1 - (input_ids == self.sep_token_id).float()
        sep_token_pos = ids.argmin(dim=-1).unsqueeze(-1)

        # mask the tokens after this position
        ids = torch.arange(0, input_ids.size(-1), device=input_ids.device)
        m = ids > sep_token_pos
        input_ids = input_ids.masked_fill(m, self.pad_token_id)
        attention_mask = attention_mask.masked_fill(m, 0)

        return {
            f"{self.field}.input_ids": input_ids,
            f"{self.field}.attention_mask": attention_mask,
        }


class LanguageModellingTransform(Pipe):
    """Language modeling task"""

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        question_field: str = "question",
        answer_field: str = "answer",
        document_field: str = "document",
        output_field: str = "lm",
        multi_doc: bool = False,
        tokenizer_kwargs: Optional[Dict] = None,
        update: bool = True,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.question_field = question_field
        self.answer_field = answer_field
        self.document_field = document_field
        self.output_field = output_field
        self.multi_doc = multi_doc
        self.input_keys = [
            f"{self.question_field}.text",
            f"{self.answer_field}.text",
            f"{self.answer_field}.target",
            f"{self.document_field}.text",
        ]

        # tokenizer kwargs
        tokenizer_kwargs_ = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "return_token_type_ids": True,
        }
        if tokenizer_kwargs is not None:
            tokenizer_kwargs_ = {**tokenizer_kwargs_, **tokenizer_kwargs}
        self.tokenizer_kwargs = tokenizer_kwargs_

        super().__init__(input_filter=In(self.input_keys), update=update, **kwargs)

    @torch.no_grad()
    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:
        eg = {k: v[0] for k, v in batch.items()}
        scenario: Scenario = infer_scenario(eg)
        if scenario == Scenario.multiple_choice_concat_qa:
            batch.pop(f"{self.answer_field}.target", None)

        # expand the batch to include all the documents
        batch = self._expand_question_or_flatten_docs(batch)

        # flatten the batch
        question_shape = infer_shape(batch[f"{self.question_field}.text"])
        flatten_pipe = Flatten(level=len(question_shape) - 1)
        batch = flatten_pipe(batch)

        eg = {k: v[0] for k, v in batch.items()}
        scenario: Scenario = infer_scenario(eg)
        template_generator = LanguageModellingTemplateGenerator(
            scenario=scenario, question_field=self.question_field, answer_field=self.answer_field
        )

        # convert the question, answer and document texts into a single string
        # using the `template_generator`.
        templates = [template_generator(eg) for eg in iter_batch_egs(batch)]

        # tokenize
        questions, answers = zip(*[(t.question, t.answer) for t in templates])
        if all(a is None for a in answers):
            tokenizer_args = (questions,)
        else:
            answers = [f"{a}{self.tokenizer.eos_token}" for a in answers]
            tokenizer_args = (questions, answers)

        output = self.tokenizer(*tokenizer_args, **self.tokenizer_kwargs)
        # for x, m in zip(output.input_ids, output.token_type_ids):
        #     rich.print(f">> (L={len(x[m > 0])}) "
        #                f"[cyan]`{self.tokenizer.decode(x[m > 0], skip_special_tokens=False)}`")
        output = {f"{self.output_field}.{k}": v for k, v in output.items()}
        output[f"{self.output_field}.text"] = [t.text for t in templates]

        # reshape the output
        output = Nest(shape=question_shape)(output)

        return output

    @staticmethod
    def join_documents(documents: List[str]) -> str:
        return " (...) ".join(documents)

    def _expand_question_or_flatten_docs(self, batch: Batch) -> Batch:
        """
        This method makes sure that the questions and documents have the same shape.
        This handles the case where multiple documents are joined into a single one,
        and this other case where the question is repeated for each document.
        """
        question_shape = infer_shape(batch[f"{self.question_field}.text"])
        document_shape = infer_shape(batch[f"{self.document_field}.text"])

        # handle the answer target key
        if f"{self.answer_field}.target" in batch:
            answer_target_shape = infer_shape(batch[f"{self.answer_field}.target"])
            if len(answer_target_shape) < len(question_shape):
                answer_target = batch.pop(f"{self.answer_field}.target", None)
            else:
                answer_target = None
        else:
            answer_target = None

        if len(question_shape) == len(document_shape):
            # the question and document are already of the same shape
            return batch
        elif len(question_shape) != len(document_shape) - 1:
            raise ValueError(
                f"Question shape {question_shape} is not "
                f"compatible with document shape {document_shape}"
            )

        else:
            # flatten the batch
            flatten_pipe = Flatten(level=len(question_shape) - 1)
            batch = flatten_pipe(batch)
            if self.multi_doc:
                # join the documents
                target_shape = question_shape
                batch[f"{self.document_field}.text"] = [
                    self.join_documents(doc) for doc in batch[f"{self.document_field}.text"]
                ]
            else:
                # expand the question
                target_shape = document_shape
                for key in [
                    f"{self.question_field}.text",
                    f"{self.answer_field}.text",
                    f"{self.answer_field}.target",
                ]:
                    if key in batch:
                        batch[key] = [[x] * document_shape[-1] for x in batch[key]]

            # reshape the first dimension
            if len(target_shape) > 2:
                batch = Nest(shape=target_shape[:-1])(batch)
            if answer_target is not None:
                batch[f"{self.answer_field}.target"] = answer_target

            return batch
