import functools
import io
from typing import Any

import torch
import xxhash
from datasets.fingerprint import Hasher
from torch import nn

from wandb.util import np


@functools.singledispatch
def get_fingerprint(obj: Any) -> str:
    """Compute an object fingerprint"""
    hash = Hasher()
    if isinstance(obj, torch.Tensor):
        obj = serialize_tensor(obj)
    hash.update(obj)
    return hash.hexdigest()


def serialize_tensor(x: torch.Tensor):
    x = x.cpu()
    buff = io.BytesIO()
    torch.save(x, buff)
    buff.seek(0)
    return buff.read()


@get_fingerprint.register(nn.Module)
def get_module_weights_fingerprint(obj: nn.Module) -> str:
    hasher = xxhash.xxh64()
    state = obj.state_dict()
    for (k, v) in sorted(state.items(), key=lambda x: x[0]):
        # it did not work without hashing the tensor
        hasher.update(k)
        u = serialize_tensor(v)
        hasher.update(u)

    return hasher.hexdigest()


def fingerprint_bert(bert):
    """Fingerprint BERT weights and the image of a random input."""
    bert_params = {k: get_fingerprint(v) for k, v in bert.named_parameters() if "encoder." in k}
    bert_fingerprint = get_fingerprint(bert_params)
    is_training = bert.training
    bert.eval()
    state = np.random.RandomState(0)
    x = state.randint(0, bert.config.vocab_size - 1, size=(3, 512))
    x = torch.from_numpy(x)
    h = bert(x).last_hidden_state
    input_fingerprint = get_fingerprint(x)
    output_fingerprint = get_fingerprint(h)
    if is_training:
        bert.train()

    return {
        "bert_weights": bert_fingerprint,
        "input_tensor": input_fingerprint,
        "ber_output_tensor": output_fingerprint,
    }
