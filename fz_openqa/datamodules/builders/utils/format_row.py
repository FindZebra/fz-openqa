from typing import Literal
from typing import Optional

import torch
from transformers import PreTrainedTokenizerFast
from warp_pipes import Eg
from warp_pipes.support.pretty import pretty_decode

ReprScenario = Literal["none", "generative-qa", "multiple-choice-qa", "multiple-choice-concat-qa"]


def infer_scenario(eg: Eg) -> ReprScenario:
    q_input_ids = eg.get("question.input_ids", None)
    a_input_ids = eg.get("answer.input_ids", None)

    # handle the case where no question is provided
    if q_input_ids is None:
        raise ValueError(
            f"No question provided (with key 'question.input_ids'). Found keys: {eg.keys()}"
        )

    # handle the case where no answer is provided
    if a_input_ids is None:
        return "generative-qa"

    # infer shapes
    q_dim = q_input_ids.ndim
    a_dim = a_input_ids.ndim

    if q_dim == 1 and a_dim == 1:
        return "generative-qa"
    elif q_dim == 2 and a_dim == 2:
        return "multiple-choice-concat-qa"
    elif q_dim == 1 and a_dim == 2:
        return "multiple-choice-qa"
    else:
        raise ValueError(f"Unknown scenario: q_dim={q_dim}, a_dim={a_dim}")


def format_qa_eg(eg: Eg, tokenizer: PreTrainedTokenizerFast, **kwargs) -> str:
    scenario = infer_scenario(eg)
    if scenario == "none":
        return f"<couldn't represent example {type(eg)}>"
    elif scenario == "generative-qa":
        return format_generative_qa_eg(eg, tokenizer, **kwargs)
    elif scenario == "multiple-choice-qa":
        return format_multiple_choice_qa_eg(eg, tokenizer, **kwargs)
    elif scenario == "multiple-choice-concat-qa":
        return format_multiple_choice_concat_qa_eg(eg, tokenizer, **kwargs)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


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
            canvas += (
                f"\nDocuments {doc_info}:{format_documents(d_tokens, tokenizer, scores=d_scores)}"
            )

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
