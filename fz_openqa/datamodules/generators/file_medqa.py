import json

import datasets


class MedQAConfig(datasets.BuilderConfig):
    """BuilderConfig for MedQAxCorpus."""

    def __init__(self, **kwargs):
        """BuilderConfig for FZxMedQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MedQAConfig, self).__init__(**kwargs)


_DESCRIPTION = "A generic class to handle MedQA datasets"
_VERSION = "0.0.1"
_HOMEPAGE = "https://github.com/MotzWanted/Open-Domain-MedQA"
_CITATION = ""


class OfficialMedQAGenerator(datasets.GeneratorBasedBuilder):
    """MedQADataset Dataset. Version 0.0.1"""

    VERSION = datasets.Version(_VERSION)
    force = False

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "question.idx": datasets.Value("int32"),
                    "question.text": datasets.Value("string"),
                    "answer.target": datasets.Value("int32"),
                    "answer.text": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        assert (
            self.config.data_files is not None
        ), "`data_files` must be provided in `load_dataset()`"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={},
            )
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        with open(filepath, "r") as f:
            for i, d in enumerate(json.load(f)["data"]):
                # adjust values
                d["question.idx"] = d.pop("question_id")
                d["answer.target"] = d.pop("answer_idx")
                d["answer.text"] = d.pop("answer_options")
                d["question.text"] = d.pop("question")
                yield i, d
