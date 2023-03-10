from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast

from .static import ADDITIONAL_SPECIAL_TOKENS
from .static import ANS_TOKEN
from .static import DOC_TOKEN
from .static import QUERY_MASK
from .static import QUERY_TOKEN

SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
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
    *, pretrained_model_name_or_path: str, **kwargs
) -> PreTrainedTokenizerFast:
    """Load a HuggingFace Pretrained Tokenizer and add the special tokens."""
    pretrained_model_name_or_path = TOKENIZERS_MAPPING.get(
        pretrained_model_name_or_path, pretrained_model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.sanitize_special_tokens()
    # test_special_token_encoding(tokenizer)
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
