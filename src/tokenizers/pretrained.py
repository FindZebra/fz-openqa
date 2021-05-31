from transformers import AutoTokenizer, PreTrainedTokenizerFast


def init_pretrained_tokenizer(
    *, pretrained_model_name_or_path: str, **kwargs
) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer
