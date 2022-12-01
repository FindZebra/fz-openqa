from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast
from warp_pipes import Eg
from warp_pipes.support.pretty import pretty_decode
from warp_pipes.support.shapes import infer_shape

from fz_openqa.datamodules.utils.datastruct import Scenario


class TextField(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    input_ids: Optional[torch.Tensor] = None
    text: Optional[Union[str, List[str]]] = None

    @property
    def batch_shape(self) -> Optional[Tuple[int, ...]]:
        if not self.exists:
            return None

        if self.input_ids is not None:
            return self.input_ids.shape[:-1]

        if self.text is not None:
            return infer_shape(self.text)

    @classmethod
    def from_field(cls, field: str, data: Dict):
        prefix = f"{field}."
        data_field = {
            str(k)[len(prefix) :]: v for k, v in data.items() if str(k).startswith(prefix)
        }
        return cls(**data_field)

    @property
    def exists(self) -> bool:
        return self.input_ids is not None or self.text is not None

    def pretty_decode(
        self, key: Optional[int | slice | Tuple] = None, style: str = "deep_sky_blue3", **kwargs
    ) -> str:
        if self.input_ids is not None:
            input_ids = self.input_ids
            if key is not None:
                input_ids = input_ids[key]
            return pretty_decode(input_ids, style=style, **kwargs)
        elif self.text is not None:
            text = self.text
            if key is not None:
                text = text[key]
            return f"[{style}]`{text}`[/{style}] [gray]<raw text>[/gray]"
        else:
            raise ValueError("No text or input_ids to decode")


def infer_scenario(eg: Eg) -> Scenario:
    def infer_field_dim(eg: Eg, field: str) -> Optional[int]:
        if f"{field}.input_ids" in eg:
            return len(infer_shape(eg[f"{field}.input_ids"]))
        elif f"{field}.text" in eg:
            return 1 + len(infer_shape(eg[f"{field}.text"]))
        else:
            return None

    q_dim = infer_field_dim(eg, "question")
    a_dim = infer_field_dim(eg, "answer")
    target = eg.get("answer.target", None)

    # handle the case where no question is provided
    if q_dim is None:
        raise ValueError(
            f"No question provided (tacking keys 'question.input_ids' and 'question.text'). "
            f"Found keys: {eg.keys()}"
        )

    # handle the case where no answer is provided
    if a_dim is None and target is None:
        return Scenario.generative_qa
    if q_dim == 1 and a_dim == 1:
        return Scenario.generative_qa
    elif q_dim == 2 and a_dim == 2:
        return Scenario.multiple_choice_concat_qa
    elif q_dim == 1 and isinstance(target, torch.Tensor) and target.dtype == torch.bool:
        return Scenario.multiple_choice_flat_concat_qa
    elif q_dim == 1 and a_dim == 2:
        return Scenario.multiple_choice_qa
    else:
        raise ValueError(f"Unknown scenario: q_dim={q_dim}, a_dim={a_dim}")


def format_qa_eg(eg: Eg, tokenizer: PreTrainedTokenizerFast, **kwargs) -> str:
    scenario = infer_scenario(eg)
    if scenario == "none":
        canvas = f"<couldn't represent example {type(eg)}>"
    elif scenario == Scenario.generative_qa:
        canvas = format_generative_qa_eg(eg, tokenizer, **kwargs)
    elif scenario == Scenario.multiple_choice_qa:
        canvas = format_multiple_choice_qa_eg(eg, tokenizer, **kwargs)
    elif scenario == Scenario.multiple_choice_concat_qa:
        canvas = format_multiple_choice_concat_qa_eg(eg, tokenizer, **kwargs)
    elif scenario == Scenario.multiple_choice_flat_concat_qa:
        canvas = format_multiple_choice_flat_concat_qa_eg(eg, tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return canvas


def format_generative_qa_eg(eg, tokenizer, **kwargs):
    question = TextField.from_field("question", eg)
    question_str = question.pretty_decode(style="deep_sky_blue3", tokenizer=tokenizer)
    canvas = f"Question: {question_str}"

    # answers
    answer = TextField.from_field("question", eg)
    if answer.exists:
        answer_str = answer.pretty_decode(style="green", tokenizer=tokenizer)
        canvas += f"\nAnswer: {answer_str}"

    # documents
    documents = TextField.from_field("document", eg)
    if documents.exists:
        d_scores = eg.get("document.proposal_score", None)
        doc_info = f"(n={documents.batch_shape[0]})"
        canvas += (
            f"\nDocuments {doc_info}:{format_documents(documents, tokenizer, scores=d_scores)}"
        )

    return canvas


def format_multiple_choice_qa_eg(eg, tokenizer, **kwargs):
    question = TextField.from_field("question", eg)
    question_str = question.pretty_decode(style="white", tokenizer=tokenizer)
    canvas = f"Question: {question_str}"

    answer = TextField.from_field("answer", eg)
    target = eg.get("answer.target", -1)
    if answer.exists:
        n_opts = answer.batch_shape[0]
        canvas += "\nAnswer options:"
        for i in range(n_opts):
            style = "green" if i == target else "deep_sky_blue3"
            marker = "✓" if i == target else " "
            answer_str = answer.pretty_decode(
                key=i, style=style, tokenizer=tokenizer, only_text=True
            )
            canvas += f"\n({marker}) {answer_str}"

    # documents
    documents = TextField.from_field("document", eg)
    if documents.exists:
        d_scores = eg.get("document.proposal_score", None)
        doc_info = f"(n={documents.batch_shape[0]})"
        canvas += (
            f"\nDocuments {doc_info}:{format_documents(documents, tokenizer, scores=d_scores)}"
        )

    return canvas


def format_multiple_choice_concat_qa_eg(eg, tokenizer, **kwargs):
    q_tokens = eg["question.input_ids"]
    d_tokens = eg.get("document.input_ids", None)
    d_scores = eg.get("document.proposal_score", None)
    target = eg.get("answer.target", -1)
    sep = "\n"
    canvas = "Question-answer pairs:"
    for i in range(q_tokens.shape[0]):
        style = "green" if i == target else "deep_sky_blue3"
        marker = "✓" if i == target else " "
        answer_str = pretty_decode(q_tokens[i], style=style, tokenizer=tokenizer, only_text=False)
        canvas += f"{sep}({marker}) {answer_str}"

        # documents
        if d_tokens is not None:
            d_scores_i = d_scores[i] if d_scores is not None else None
            canvas += rf"{format_documents(d_tokens[i], tokenizer, scores=d_scores_i)}"

    return canvas


def format_multiple_choice_flat_concat_qa_eg(eg, tokenizer, **kwargs):
    q_tokens = eg["question.input_ids"]
    target = eg.get("answer.target", False)

    sep = "\n"
    canvas = "Question-answer pair:"
    style = "green" if target else "deep_sky_blue3"
    marker = "✓" if target else " "
    q_str = pretty_decode(q_tokens, style=style, tokenizer=tokenizer, only_text=False)
    canvas += f"{sep}({marker}) {q_str}"

    # documents
    d_tokens = eg.get("document.input_ids", None)
    d_scores = eg.get("document.proposal_score", None)
    if d_tokens is not None:
        doc_info = f"(n={len(d_tokens)})"
        canvas += f"\nDocuments {doc_info}:{format_documents(d_tokens, tokenizer, scores=d_scores)}"

    return canvas


def format_documents(
    documents: TextField,
    tokenizer: PreTrainedTokenizerFast,
    style="white",
    scores: Optional[torch.Tensor] = None,
    top_k: int = 3,
    sep="\n  Doc",
    **kwargs,
):
    def fmt_score(scores, i):
        if scores is None:
            return ""
        else:
            return f", {scores[i]:.2f}"

    canvas = ""
    for i in range(documents.batch_shape[0])[:top_k]:
        doc_str = documents.pretty_decode(key=i, style=style, tokenizer=tokenizer, only_text=True)
        loc = f"(#{i + 1}{fmt_score(scores, i)})"
        canvas += f"{sep}{loc} {doc_str}"
    return canvas
