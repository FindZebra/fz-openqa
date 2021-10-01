from unittest import TestCase
from transformers import AutoTokenizer

class TestDynamicRanking(TestCase):
    def setUp(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
