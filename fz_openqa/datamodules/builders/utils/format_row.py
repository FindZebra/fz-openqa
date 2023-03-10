from copy import copy
from typing import Any
from typing import Callable
from typing import Dict

import torch
from warp_pipes import get_console_separator
from warp_pipes.support.pretty import pretty_decode
from warp_pipes.support.shapes import infer_shape


def format_row_flat_questions(row: Dict[str, Any], *, tokenizer, **kwargs) -> str:
    decode_kwargs = {
        "skip_special_tokens": False,
        "tokenizer": tokenizer,
    }
    u = "* Question:"
    u += (
        pretty_decode(
            row["question.input_ids"],
            **decode_kwargs,
            style="deep_sky_blue3",
        )
        + "\n"
    )

    if "answer.input_ids" in row:
        u += get_console_separator("-") + "\n"
        u += "* Answer Choices:" + "\n"
        idx = row.get("answer.target", None)
        for i, an in enumerate(row["answer.input_ids"]):
            an_style = "green" if idx == i else "white"
            line = f"   - ({'x' if idx == i else ' '}) " if idx is not None else "   - "
            line += f"{pretty_decode(an, **decode_kwargs, only_text=True, style=an_style)}\n"
            u += line

    # if "document.input_ids" in row:
    #     u += "* Document:" + "\n"
    #     u += (
    #             pretty_decode(
    #                 row["document.input_ids"],
    #                 **decode_kwargs,
    #                 style="white",
    #             )
    #             + "\n"
    #     )

    return u


def format_row_concatenated_questions(row: Dict[str, Any], *, tokenizer, **kwargs):
    """Decode and print one row from the batch

    Parameters
    ----------
    **kwargs
    """
    decode_kwargs = {
        "skip_special_tokens": False,
        "tokenizer": tokenizer,
    }
    repr = f"Question #{row.get('question.idx', None)}\n"

    repr += get_console_separator("-") + "\n"
    repr += "* Question + answer:" + "\n"
    idx = row.get("answer.target", None)
    for i, an in enumerate(row["question.input_ids"]):
        an_style = "green" if idx == i else "white"
        line = ""
        if idx is not None:
            line += f"   - ({'x' if idx == i else ' '}) "
        else:
            line += " - "
        line += f"{pretty_decode(an, **decode_kwargs, only_text=False, style=an_style)}\n"
        repr += line

    return repr


def get(row: Dict[str, Any], key: str, *i: int, default=None) -> Any:
    if key not in row:
        return default
    x = row[key]
    for j in i:
        x = x[j]
    return x


def format_row_flat_questions_with_docs(
    row: Dict[str, Any],
    *,
    tokenizer,
    format_question_fn: Callable,
    max_documents: int = 3,
    **kwargs,
) -> str:
    decode_kwargs = {
        "skip_special_tokens": False,
        "tokenizer": tokenizer,
    }

    repr = format_question_fn(row, tokenizer=tokenizer, **kwargs)
    repr += get_console_separator("-") + "\n"
    repr += f"* Documents: n={len(row['document.input_ids'])}"
    if "document.match_score" in row:
        repr += f", n_positive={sum(row['document.match_score'] > 0)}, "
        f"n_negative={sum(row['document.match_score'] == 0)}\n"
    repr += "\n"
    for j in range(min(len(row["document.input_ids"]), max_documents)):
        repr += get_console_separator(".") + "\n"
        match_on = row.get("document.match_on", None)
        match_on = match_on[j] if match_on is not None else None
        match_score = row.get("document.match_score", None)
        match_score = match_score[j] if match_score is not None else None
        repr += (
            f"|-* Document #{1 + j}, "
            f"score={row['document.proposal_score'][j]:.2f}, "
            f"match_score={match_score}, "
            f"match_on={match_on}\n"
        )

        repr += (
            pretty_decode(
                row["document.input_ids"][j],
                **decode_kwargs,
                style="white",
            )
            + "\n"
        )

    return repr


def format_row_nested_questions_with_docs(
    row: Dict[str, Any], *, tokenizer, document_nesting_level: int, max_documents: int = 5, **kwargs
) -> str:
    decode_kwargs = {
        "skip_special_tokens": False,
        "tokenizer": tokenizer,
    }
    row = copy(row)

    repr = f"* Question #{row.get('question.idx', None)}\n"
    idx = row["answer.target"]

    # for each question-answer pair
    for i, an in enumerate(row["question.input_ids"]):
        locator = f"Option #{i + 1} (Q#{row.get('question.idx', None)})"
        repr += get_console_separator("-") + "\n"
        repr += f"|-* {locator}\n"
        # print question-answer pair
        an_style = "green" if idx == i else "cyan"
        line = (
            f"   - ({'x' if idx == i else ' '}) "
            f"{pretty_decode(an, **decode_kwargs, only_text=False, style=an_style)}\n"
        )
        repr += line

        if document_nesting_level == 2:
            # print documents attached to the question-answer pair
            document_row = {k: v[i] for k, v in row.items() if k.startswith("document.")}
            repr += repr_documents(
                document_row, locator, **decode_kwargs, max_documents=max_documents
            )

    if document_nesting_level == 1:
        repr += repr_documents(row, "", **decode_kwargs)

    return repr


def repr_documents(row, locator, max_documents: int = 3, **decode_kwargs) -> str:
    """represent a row of documents"""
    repr = ""
    repr += get_console_separator(".") + "\n"
    repr += f"|-* Q#{locator} - Documents: n={len(row['document.input_ids'])}"
    if "document.match_score" in row:
        repr += (
            f", n_positive={sum(row['document.match_score'] > 0)}, "
            f"n_negative={sum(row['document.match_score'] == 0)}"
        )
    repr += "\n"
    # for each document
    for j in range(min(len(row["document.input_ids"]), max_documents)):
        match_on = row.get("document.match_on", None)
        match_on = match_on[j] if match_on is not None else None
        repr += f"|---* Q#{locator} - Document #{1 + j} "
        repr += f"(id={get(row, 'document.idx', j)}, row_idx={get(row, 'document.row_idx', j)}), "
        repr += f"score={row['document.proposal_score'][j]:.2f}, "
        if "document.match_score" in row:
            repr += f"match_score={row['document.match_score'][j]}, " f"match_on={match_on}"
        if "document.question_idx" in row:
            repr += f", question_idx={get(row, 'document.question_idx', j)}"
        repr += "\n"

        doc_style = "yellow" if match_on else "white"
        repr += (
            pretty_decode(
                row["document.input_ids"][j],
                **decode_kwargs,
                style=doc_style,
            )
            + "\n"
        )
    return repr


def format_row_qa(row: Dict[str, Any], *, tokenizer, **kwargs):
    """Decode and print one row from the batch"""
    decode_kwargs = {
        "skip_special_tokens": False,
        "tokenizer": tokenizer,
    }
    q_input_ids = row["question.input_ids"]
    q_input_ids_shape = infer_shape(q_input_ids)
    nesting_level = len(q_input_ids_shape) - 1
    qid = row.get("question.idx", None)
    if qid is None:
        qid = row.get("question.id", None)
    q_label = row.get("answer.target", None)
    if q_label is not None:
        q_label_shape = infer_shape(q_label)
        if len(q_label_shape) < nesting_level:
            q_label_binary = torch.zeros_like(q_label, dtype=torch.bool)
            q_label_binary = q_label_binary.expand(q_input_ids_shape[:-1]).clone()
            q_label_binary.scatter_(dim=-1, index=q_label, value=1)
            q_label = q_label_binary
            row["answer.target"] = q_label

    repr = ""
    if nesting_level == 0:
        # repr question
        repr += get_console_separator("-") + "\n"
        repr += f"* Question #{qid}:" + "\n"
        an_style = "green" if q_label else "white"
        line = ""
        if q_label is not None:
            line += f" - ({'x' if q_label else ' '}) "
        else:
            line += " - "
        line += f"{pretty_decode(q_input_ids, **decode_kwargs, only_text=False, style=an_style)}\n"
        repr += line

        # add documents repr
        if "document.input_ids" in row:
            repr += repr_documents(row, qid, **decode_kwargs)
    else:
        for i in range(q_input_ids_shape[0]):
            row_i = {k: v[i] for k, v in row.items() if len(infer_shape(v)) > 0}
            repr += format_row_qa(row_i, tokenizer=tokenizer, **kwargs)

    return repr
