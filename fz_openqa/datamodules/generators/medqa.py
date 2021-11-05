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


_TRAIN_URL = "https://drive.google.com/file/d/18a1TxYHHlNqXNBHaSgfLRgI4kBmYXcyn/view?usp=sharing"
_VALID_URL = "https://drive.google.com/file/d/1m4zUJoET3WDqpYvQ_aOJVmJbiSjGGhB0/view?usp=sharing"
_TEST_URL = "https://drive.google.com/file/d/1cOOSjOjBIOlzi3Wk31kxnp-eIV6Ekslh/view?usp=sharing"

_DESCRIPTION = "A mapping between the MedQA dataset and the MedQA corpus (18 books)"
_VERSION = "0.0.1"
_HOMEPAGE = ""
_CITATION = ""


class MedQaGenerator(datasets.GeneratorBasedBuilder):
    """MedQAxCorpus Dataset. Version 0.0.1"""

    VERSION = datasets.Version(_VERSION)
    force = False
    urls = {
        datasets.Split.TRAIN: _TRAIN_URL,
        datasets.Split.VALIDATION: _VALID_URL,
        datasets.Split.TEST: _TEST_URL,
    }

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "question.idx": datasets.Value("int32"),
                    "question.text": datasets.Value("string"),
                    "question.metamap": datasets.Sequence(datasets.Value("string")),
                    "answer.target": datasets.Value("int32"),
                    "answer.text": datasets.Sequence(datasets.Value("string")),
                    "answer.cui": datasets.Sequence(datasets.Value("string")),
                    "answer.synonyms": datasets.Sequence(datasets.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    @staticmethod
    def _get_drive_url(url):
        base_url = "https://drive.google.com/uc?id="
        split_url = url.split("/")
        return base_url + split_url[5]

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = {
            split: dl_manager.download_and_extract(self._get_drive_url(url))
            for split, url in self.urls.items()
        }

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"filepath": file, "split": split},
            )
            for split, file in downloaded_files.items()
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        with open(filepath, "r") as f:
            for i, d in enumerate(json.load(f)["data"]):
                # adjust values
                d["question.idx"] = d.pop("question_id")
                d["answer.target"] = d.pop("answer_idx")
                d["answer.text"] = d.pop("answer_options")
                d["answer.cui"] = d.pop("CUIs", None)
                d["answer.synonyms"] = d.pop("synonyms", None)
                d["question.text"] = d.pop("question")
                d["question.metamap"] = d.pop("question_filt", None)
                yield i, d
