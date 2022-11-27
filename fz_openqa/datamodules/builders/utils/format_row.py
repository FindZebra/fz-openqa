from enum import Enum
from typing import Optional

import torch
from transformers import PreTrainedTokenizerFast
from warp_pipes import Eg
from warp_pipes.support.pretty import pretty_decode
from warp_pipes.support.shapes import infer_shape


class Scenario(Enum):
    none = "none"
    generative_qa = "generative-qa"
    multiple_choice_qa = "multiple-choice-qa"
    multiple_choice_concat_qa = "multiple-choice-concat-qa"
    multiple_choice_flat_concat_qa = "multiple-choice-flat_concat-qa"


def infer_field_dim(eg: Eg, field: str) -> Optional[int]:
    if f"{field}.input_ids" in eg:
        return len(infer_shape(eg[f"{field}.input_ids"]))
    elif f"{field}.text" in eg:
        return 1 + len(infer_shape(eg[f"{field}.text"]))
    else:
        return None


def infer_scenario(eg: Eg) -> Scenario:
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
    question_str = pretty_decode(
        eg["question.input_ids"], style="deep_sky_blue3", tokenizer=tokenizer
    )
    canvas = f"Question: {question_str}"

    # answers
    a_tokens = eg.get("answer.input_ids", None)
    if a_tokens is not None:
        answer_str = pretty_decode(a_tokens, style="green", tokenizer=tokenizer)
        canvas += f"\nAnswer: {answer_str}"

    # documents
    d_tokens = eg.get("document.input_ids", None)
    d_scores = eg.get("document.proposal_score", None)
    if d_tokens is not None:
        doc_info = f"(n={len(d_tokens)})"
        canvas += f"\nDocuments {doc_info}:{format_documents(d_tokens, tokenizer, scores=d_scores)}"

    return canvas


def format_multiple_choice_qa_eg(eg, tokenizer, **kwargs):
    question_str = pretty_decode(eg["question.input_ids"], style="white", tokenizer=tokenizer)
    canvas = f"Question: {question_str}"

    a_tokens = eg.get("answer.input_ids", None)
    target = eg.get("answer.target", -1)
    if a_tokens is not None:
        n_opts = a_tokens.shape[0]
        canvas += "\nAnswer options:"
        for i in range(n_opts):
            style = "green" if i == target else "deep_sky_blue3"
            marker = "✓" if i == target else " "
            answer_str = pretty_decode(
                a_tokens[i], style=style, tokenizer=tokenizer, only_text=True
            )
            canvas += f"\n({marker}) {answer_str}"

    # documents
    d_tokens = eg.get("document.input_ids", None)
    d_scores = eg.get("document.proposal_score", None)
    if d_tokens is not None:
        doc_info = f"(n={len(d_tokens)})"
        canvas += f"\nDocuments {doc_info}:{format_documents(d_tokens, tokenizer, scores=d_scores)}"

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
    doc_tokens: torch.Tensor,
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
    for i in range(doc_tokens.shape[0])[:top_k]:
        doc_str = pretty_decode(doc_tokens[i], style=style, tokenizer=tokenizer, only_text=True)
        loc = f"(#{i + 1}{fmt_score(scores, i)})"
        canvas += f"{sep}{loc} {doc_str}"
    return canvas
