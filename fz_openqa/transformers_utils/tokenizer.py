import string
from typing import Optional

import loguru
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import BertNormalizer
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast

QUERY_TOKEN = "<|QUERY|>"
QUERY_MASK = "<|QMASK|>"
DOC_TOKEN = "<|DOC|>"
ANS_TOKEN = "<|ANS|>"
TRUNCATED_TOKEN = "<|TRUNCATED|>"

ADDITIONAL_SPECIAL_TOKENS = [QUERY_TOKEN, DOC_TOKEN, ANS_TOKEN, QUERY_MASK, TRUNCATED_TOKEN]

SPECIAL_TOKENS = {
    "pad_token": "[PAD] ",
    "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS,
}

CHAR_SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "cls_token": "[CLS]",
    "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS,
}

# lookup the base version of the tokenizer name to avoid duplicating dataset preprocessing
TOKENIZERS_MAPPING = {
    "bert-large-cased": "bert-base-cased",
    "bert-large-uncased": "bert-base-uncased",
    "sultan/BioM-ELECTRA-Large-Discriminator": "sultan/BioM-ELECTRA-Base-Discriminator",
    "microsoft/deberta-v3-large": "microsoft/deberta-v3-base",
    "dmis-lab/biobert-large-cased-v1.2": "dmis-lab/biobert-base-cased-v1.2",
    "michiyasunaga/LinkBERT-large": "michiyasunaga/LinkBERT-base",
    "michiyasunaga/BioLinkBERT-large": "michiyasunaga/BioLinkBERT-base",
    "microsoft/BiomedNLP-PubMedBERT-"
    "large-uncased-abstract-fulltext": "microsoft/BiomedNLP-PubMedBERT-"
    "base-uncased-abstract-fulltext",
}


def init_pretrained_tokenizer(
    *,
    model_id: str,
    add_special_tokens: bool = True,
    set_padding_side: Optional[str] = None,
    pad_token: Optional[str] = None,
    **kwargs,
) -> PreTrainedTokenizerFast:
    """Load a HuggingFace Pretrained Tokenizer and add the special tokens."""
    model_id = TOKENIZERS_MAPPING.get(model_id, model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if add_special_tokens:
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
    if pad_token is not None:
        if pad_token == "tokenizer.eos_token":
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = pad_token
    tokenizer.sanitize_special_tokens()
    if set_padding_side is not None:
        loguru.logger.info(f"Setting tokenizer padding size to `{set_padding_side}`")
        tokenizer.padding_side = set_padding_side
    return tokenizer


def init_char_tokenizer() -> PreTrainedTokenizerFast:
    # initialize tokenizer as Byte Pair Encoder
    # it won't be trained as a BPE, as we set the vocabulary,
    # but this allows benefiting from the all the Tokenizer methods
    tokenizer = Tokenizer(BPE())

    # define vocabulary and special tokens
    vocab = list(string.printable)

    # get normalizer
    tokenizer.normalizer = BertNormalizer()

    # set the vocabulary the vocabulary
    tokenizer.add_tokens(vocab)

    # convert as PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    # add the special tokens
    tokenizer.add_special_tokens(CHAR_SPECIAL_TOKENS)
    tokenizer.sanitize_special_tokens()

    return tokenizer
