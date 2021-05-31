import os
import string

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import BertNormalizer
from transformers import PreTrainedTokenizerFast


def init_char_tokenizer() -> PreTrainedTokenizerFast:
    # initialize tokenizer as Byte Pair Encoder
    # it won't be trained as a BPE, as we set the vocabulary,
    # but this allows benefiting from the all the Tokenizer methods
    tokenizer = Tokenizer(BPE())

    # define vocabulary and special tokens
    vocab = list(string.printable)
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    # get normalizer
    tokenizer.normalizer = BertNormalizer()

    # train the BPE to learn the vocabulary
    tokenizer.add_tokens(special_tokens + vocab)

    return PreTrainedTokenizerFast(tokenizer_object=tokenizer)


tokenizer = init_char_tokenizer()

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

print(fast_tokenizer.encode_plus(f"Hello world!"))