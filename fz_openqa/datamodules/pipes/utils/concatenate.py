import warnings

import einops
import torch

from fz_openqa.modeling.functional import padless_cat
from fz_openqa.utils.datastruct import Batch


def concat_questions_and_documents(batch: Batch, *, pad_token_id: int, max_length: int) -> Batch:
    """
    Concatenate the questions and the documents across the time dimension, and without padding.
    Parameters
    ----------
    batch : Batch
        The batch to concatenate, with fields `question` and `document`:
        - `question_*`: (batch_size, n_opts, q_length)
        - `document_*`: (batch_size, n_opts, n_docs, d_length) or (batch_size, n_docs, d_length)
    pad_token_id : int
        The token id of the padding token.
    max_length : int
        The maximum length of the concatenated sequence.
    Returns
    -------
    Batch
        the concatenated fields:
            - `input_ids`: (batch_size * n_docs, n_opts, total_length)
            - `attention_mask`: (batch_size * n_docs, n_opts, total_length)
    """

    fields = ["question", "document"]

    # infer the shape of the questions, expects [bs, n_options, seq_len]
    bs, n_opts, q_length = batch["question.input_ids"].shape

    # infer the shape of the documents
    doc_shape = batch["document.input_ids"].shape
    if len(doc_shape) == 4:
        # case where there are `n_docs` documents per answer option: [bs, n_opts, n_docs, seq_len]
        pass
    elif len(doc_shape) == 3:
        # case where there are `n_docs` documents per question: [bs,n_docs, seq_len]
        # then expand the documents to shape [bs, n_opts, n_docs, seq_len]
        for key in ["document.input_ids", "document.attention_mask"]:
            batch[key] = einops.repeat(batch[key], "bs n_opts ...", n_opts=n_opts)
    else:
        raise ValueError(f"Unsupported document shape {doc_shape}. Expected 3 or 4 dimensions.")

    # infer the shape of the documents
    bs_, n_opts_, n_docs, d_length = batch["document.input_ids"].shape
    assert bs == bs_, "Batch size mismatch"
    assert n_opts == n_opts_, "Number of options mismatch"

    # flatten [bs, n_options, ...] -> [bs * n_options, ...]
    for key in [
        "document.input_ids",
        "question.input_ids",
        "document.attention_mask",
        "question.attention_mask",
    ]:
        batch[key] = einops.rearrange(batch[key], "bs n_opts ... -> (bs n_opts) ...")

    # split documents and questions, remove the cls token
    # todo: @andreas: not sure if we should remove the intermediate [SEP] and [DOC] tokens
    inputs = [
        {
            "input_ids": batch["question.input_ids"][..., 1:],
            "attention_mask": batch["question.attention_mask"][..., 1:],
        }
    ]
    for i in range(n_docs):
        inputs.append(
            {
                "input_ids": batch["document.input_ids"][:, i, 1:],
                "attention_mask": batch["document.attention_mask"][:, i, 1:],
            }
        )

    # concatenate keys across the time dimension, CLS tokens are removed
    padded_batch = padless_cat(
        inputs,
        master_key="input_ids",
        pad_token=pad_token_id,
        aux_pad_tokens={"attention_mask": 0},
    )

    # append the CLS tokens
    for key in ["input_ids", "attention_mask"]:
        cls_tokens = batch[f"question.{key}"][..., :1]
        padded_batch[key] = torch.cat([cls_tokens, padded_batch[key]], dim=-1)

    # truncate the inputs to the maximum length
    input_length = padded_batch["input_ids"].shape[-1]
    if max_length is not None and input_length > max_length:
        warnings.warn(f"the tensor [{'; '.join(fields)}] was truncated.")
        for key in ["input_ids", "attention_mask"]:
            padded_batch[key] = padded_batch[key][..., :max_length]

    # restore the original shape to [bs, n_opts, seq_lengths]
    for key in ["input_ids", "attention_mask"]:
        padded_batch[key] = einops.rearrange(
            padded_batch[key], "(bs n_opts) ... -> bs n_opts ...", bs=bs, n_opts=n_opts
        )

    return padded_batch


def stack_questions_and_documents(batch: Batch, *, pad_token_id: int, max_length: int) -> Batch:
    """
    Expand the questions to match the documents shape [bs, n_opts, n_docs, ...]
    and concatenate with the documents across the time dimension, and without padding.
    Parameters
    ----------
    batch : Batch
        The batch to concatenate, with fields `question` and `document`:
        - `question_*`: (batch_size, n_opts, q_length)
        - `document_*`: (batch_size, n_opts, n_docs, d_length) or (batch_size, n_docs, d_length)
    pad_token_id : int
        The token id of the padding token.
    max_length : int
        The maximum length of the concatenated sequence.
    Returns
    -------
    Batch
        the concatenated fields:
            - `input_ids`: (batch_size * n_docs, n_opts, total_length)
            - `attention_mask`: (batch_size * n_docs, n_opts, total_length)
    """

    fields = ["question", "document"]

    # infer the shape of the questions, expects [bs, n_options, seq_len]
    bs, n_opts, q_length = batch["question.input_ids"].shape

    # infer the shape of the documents
    doc_shape = batch["document.input_ids"].shape
    if len(doc_shape) == 4:
        # case where there are `n_docs` documents per answer option: [bs, n_opts, n_docs, seq_len]
        pass
    elif len(doc_shape) == 3:
        # case where there are `n_docs` documents per question: [bs,n_docs, seq_len]
        # then expand the documents to shape [bs, n_opts, n_docs, seq_len]
        for key in ["document.input_ids", "document.attention_mask"]:
            batch[key] = einops.repeat(batch[key], "bs n_opts ...", n_opts=n_opts)
    else:
        raise ValueError(f"Unsupported document shape {doc_shape}. Expected 3 or 4 dimensions.")

    # infer the shape of the documents
    bs_, n_opts_, n_docs, d_length = batch["document.input_ids"].shape
    assert bs == bs_, "Batch size mismatch"
    assert n_opts == n_opts_, "Number of options mismatch"

    # expand the questions to match the documents shape [bs, n_options, n_docs, ...]
    for key in ["question.input_ids", "question.attention_mask"]:
        batch[key] = einops.repeat(
            batch[key], "bs n_opts ... -> bs n_opts n_docs ...", n_docs=n_docs
        )

    # flatten [bs, n_options, ...] -> [bs * n_options, ...]
    for key in [
        "document.input_ids",
        "question.input_ids",
        "document.attention_mask",
        "question.attention_mask",
    ]:
        batch[key] = einops.rearrange(
            batch[key],
            "bs n_opts n_docs ... -> (bs n_opts n_docs) ...",
            bs=bs,
            n_docs=n_docs,
            n_opts=n_opts,
        )

    # get the list of inputs and remove the cls token
    # todo: @andreas: not sure if we should remove the intermediate [SEP] and [DOC] tokens
    inputs = [
        {
            "input_ids": batch[f"{field}.input_ids"][..., 1:],
            "attention_mask": batch[f"{field}.attention_mask"][..., 1:],
        }
        for field in fields
    ]

    # concatenate keys across the time dimension, CLS tokens are removed
    padded_batch = padless_cat(
        inputs,
        master_key="input_ids",
        pad_token=pad_token_id,
        aux_pad_tokens={"attention_mask": 0},
    )

    # append the CLS tokens
    for key in ["input_ids", "attention_mask"]:
        cls_tokens = batch[f"question.{key}"][..., :1]
        padded_batch[key] = torch.cat([cls_tokens, padded_batch[key]], dim=-1)

    # truncate the inputs to the maximum length
    input_length = padded_batch["input_ids"].shape[-1]
    if max_length is not None and input_length > max_length:
        warnings.warn(f"the tensor [{'; '.join(fields)}] was truncated.")
        for key in ["input_ids", "attention_mask"]:
            padded_batch[key] = padded_batch[key][..., :max_length]

    # restore the original shape to [bs * n_docs, n_opts, seq_lengths]
    for key in ["input_ids", "attention_mask"]:
        padded_batch[key] = einops.rearrange(
            padded_batch[key],
            "(bs n_opts n_docs) ... -> bs n_opts n_docs ...",
            bs=bs,
            n_opts=n_opts,
            n_docs=n_docs,
        )

    return padded_batch
