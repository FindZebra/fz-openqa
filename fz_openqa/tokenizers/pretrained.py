from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast

from .static import ADDITIONAL_SPECIAL_TOKENS
from .static import ANS_TOKEN
from .static import DOC_TOKEN
from .static import QUERY_TOKEN

SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS,
}


def init_pretrained_tokenizer(
    *, pretrained_model_name_or_path: str, **kwargs
) -> PreTrainedTokenizerFast:
    """Load a HuggingFace Pretrained Tokenizer and add the special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, **kwargs
    )
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.sanitize_special_tokens()
    test_special_token_encoding(tokenizer)
    return tokenizer


def test_special_token_encoding(tokenizer):
    text = "hello world!"
    for t in [QUERY_TOKEN, DOC_TOKEN, ANS_TOKEN]:
        t_id = tokenizer.get_vocab()[t]
        tokens = tokenizer(f"{t}{text}").input_ids
        assert tokens[0] == tokenizer.cls_token_id
        assert tokens[1] == t_id
        assert tokens[-1] == tokenizer.sep_token_id
        assert text == tokenizer.decode(tokens, skip_special_tokens=True)
