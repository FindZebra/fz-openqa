import string

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


def init_pretrained_tokenizer(*, model_id: str, **kwargs) -> PreTrainedTokenizerFast:
    """Load a HuggingFace Pretrained Tokenizer and add the special tokens."""
    model_id = TOKENIZERS_MAPPING.get(model_id, model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.sanitize_special_tokens()
    # test_special_token_encoding(tokenizer)
    return tokenizer


CHAR_SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "cls_token": "[CLS]",
    "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS,
}


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


def test_special_token_encoding(tokenizer: PreTrainedTokenizerFast):
    text = "hello world!"
    for t in [QUERY_TOKEN, DOC_TOKEN, ANS_TOKEN, QUERY_MASK]:
        t_id = tokenizer.get_vocab()[t]
        tokens = tokenizer(f"{t}{text}", add_special_tokens=True).input_ids
        if not tokens[0] in (tokenizer.cls_token_id, tokenizer.bos_token_id):
            decoded = tokenizer.decode([tokens[0]])
            raise ValueError(
                f"{tokens[0]}={decoded} is not {tokenizer.cls_token} or {tokenizer.bos_token}"
            )
        if not tokens[-1] in (tokenizer.sep_token_id, tokenizer.eos_token_id):
            decoded = tokenizer.decode([tokens[-1]])
            raise ValueError(
                f"{tokens[-1]}={decoded} is not {tokenizer.sep_token} or {tokenizer.eos_token}"
            )
        assert tokens[1] == t_id
        assert text == tokenizer.decode(tokens, skip_special_tokens=True)
