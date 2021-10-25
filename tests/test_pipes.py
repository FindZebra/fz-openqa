from copy import deepcopy
from dataclasses import dataclass
from unittest import TestCase

from transformers import AutoTokenizer

from fz_openqa.datamodules.pipes import Pipe, AddPrefix, DropKeys, FilterKeys, GetKey, Identity, \
    Lambda, ReplaceInKeys, RenameKeys, Apply, ApplyToAll, CopyBatch, Batchify
from fz_openqa.utils.datastruct import Batch


@dataclass
class PipeOutput:
    input: Batch
    output: Batch


class TestPipesOutputKeys(TestCase):

    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        text = [
            "Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.",
            "It also contains supporting code for evaluation and parameter tuning.",
            "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
        ]

        self._batch = {'text': text,
                       'attribute': [1, 2, 3],
                       "other": ["a", "b", "c"],
                       "question.feature_a": [5, 6, 7],
                       "question.feature_b": ["x", "y", "z"]}

    @property
    def batch(self):
        return deepcopy(self._batch)

    def _call_pipe(self, pipe: Pipe) -> PipeOutput:
        """Copy the input batch and call the pipe. Check the consistency of the output keys."""
        batch = self.batch
        inferred_keys = pipe.output_keys(list(batch.keys()))
        output = pipe(batch)
        self.assertEqual(set(inferred_keys), set(output.keys()))
        return PipeOutput(input=batch, output=output)

    def test_identity(self):
        pipe = Identity()
        self._call_pipe(pipe)

    def test_lambda(self):
        pipe = Lambda(lambda x: x)
        self._call_pipe(pipe)

        pipe = Lambda(lambda x: {'new_attr': None}, output_keys=['new_attr'])
        self._call_pipe(pipe)

    def test_get_key(self):
        pipe = GetKey("attribute")
        self._call_pipe(pipe)
        pipe = GetKey("text")
        self._call_pipe(pipe)

    def test_fitler_keys(self):
        pipe = FilterKeys(lambda key: "text" not in key)
        output = self._call_pipe(pipe)
        self.assertNotIn("text", output.output.keys())

    def test_drop_keys(self):
        pipe = DropKeys(keys=["attribute"])
        self._call_pipe(pipe)
        pipe = DropKeys(keys=["text", "other"])
        self._call_pipe(pipe)

    def test_add_prefix(self):
        pipe = AddPrefix("prefix.")
        self._call_pipe(pipe)

    def test_replace_in_key(self):
        pipe = ReplaceInKeys("question.", "document.")
        self._call_pipe(pipe)

    def test_rename_keys(self):
        pipe = RenameKeys({"text": "document", "question.feature_a": "document.feature_a"})
        self._call_pipe(pipe)

    def test_apply(self):
        pipe = Apply({'text': lambda x: x}, element_wise=False)
        self._call_pipe(pipe)
        pipe = Apply({'text': lambda x: x}, element_wise=True)
        self._call_pipe(pipe)

    def test_apply_to_all(self):
        pipe = ApplyToAll(lambda x: x)
        self._call_pipe(pipe)

    def test_copy_batch(self):
        pipe = CopyBatch()
        self._call_pipe(pipe)

    def test_batchify(self):
        pipe = Batchify()
        self._call_pipe(pipe)
