import string

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import BertNormalizer
from transformers import PreTrainedTokenizerFast
from .static import ADDITIONAL_SPECIAL_TOKENS

SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "cls_token": "[CLS]",
    "additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS
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
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.sanitize_special_tokens()

    return tokenizer
