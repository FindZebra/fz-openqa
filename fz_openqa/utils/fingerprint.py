from typing import Dict

import numpy as np
import torch
from transformers import BertModel
from warp_pipes import get_fingerprint


def fingerprint_bert(bert: BertModel) -> Dict[str, str]:
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
