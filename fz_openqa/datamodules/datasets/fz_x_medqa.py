import json

import datasets


class FZxMedQAConfig(datasets.BuilderConfig):
    """BuilderConfig for FZxMedQA."""

    def __init__(self, **kwargs):
        """BuilderConfig for FZxMedQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FZxMedQAConfig, self).__init__(**kwargs)


_TRAIN_URL = "https://drive.google.com/file/d/1-0yhF7QxAH6bWLO1Dn9V7K9pUc0VtCCA/view?usp=sharing"
_VALID_URL = "https://drive.google.com/file/d/1AAyNRxGevRj5mA7BeMsc2M5TtztN9LpZ/view?usp=sharing"
_TEST_URL = "https://drive.google.com/file/d/1kKcYiNvPs1vw9AibXmtyrFChmtnbEDXw/view?usp=sharing"

_DESCRIPTION = "A mapping between the FinzdZebra corpus and the MedQA dataset"
_VERSION = "0.0.1"
_HOMEPAGE = "https://github.com/MotzWanted/Open-Domain-MedQA"
_CITATION = ""


class FZxMedQADataset(datasets.GeneratorBasedBuilder):
    """FZxMedQA Dataset. Version 0.0.1"""

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
                    "idx": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "question.idx": datasets.Value("int32"),
                    "answer.target": datasets.Value("int32"),
                    "answer": datasets.Sequence(datasets.Value("string")),
                    "document": datasets.Value("string"),
                    "document.title": datasets.Value("string"),
                    "document.rank": datasets.Value("int32"),
                    "document.is_positive": datasets.Value("bool"),
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
                # extract document text and title
                document = d["document"]
                title, *text = document.split(". ")
                text = ". ".join(text)
                d["document"] = text
                d["document.title"] = title

                # adjust other values
                d["document.rank"] = d.pop("rank_bm25") - 1  # start from zero
                d["document.is_positive"] = d.pop("is_positive")
                d["question.idx"] = d.pop("question_id")
                d["answer.target"] = d.pop("answer_idx")
                d["answer"] = d.pop("answer_choices")
                yield i, d
