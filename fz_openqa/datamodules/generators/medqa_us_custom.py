import json
from random import Random

import datasets


class MedQAConfig(datasets.BuilderConfig):
    """BuilderConfig for MedQAxCorpus."""

    def __init__(self, **kwargs):
        """BuilderConfig for FZxMedQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MedQAConfig, self).__init__(**kwargs)


_TRAIN_URL = "https://drive.google.com/file/d/1SVK5QIjzpeEoi6LA-YPEFaOwZj2HZyqu/view?usp=sharing"
_VALID_URL = "https://drive.google.com/file/d/1pjBxHTvc5rj5VHxE3IEx4NyBBD1329u5/view?usp=sharing"
_TEST_URL = "https://drive.google.com/file/d/1W2UNjPutO2tkReFcKKtj16nyBrYZqkka/view?usp=sharing"

_DESCRIPTION = "MedQA dataset - custom 4 options extraction"
_VERSION = "0.0.1"
_HOMEPAGE = ""
_CITATION = ""


class CustomMedQaGenerator(datasets.GeneratorBasedBuilder):
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
                    "answer.target": datasets.Value("int32"),
                    "answer.text": datasets.Sequence(datasets.Value("string")),
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
            for i, line in enumerate(f.readlines()):
                d = json.loads(line)
                # get raw data
                question = d["question"]
                answer = d["answer"]
                options = list(d["options"].values())
                assert answer == d["options"][d["answer_idx"]]

                # extract 4 options
                assert len(options) == 5
                options.remove(answer)
                assert len(options) == 4
                Random(i).shuffle(options)
                options = [answer] + options[:3]
                Random(i).shuffle(options)
                assert len(options) == 4
                target = options.index(answer)
                yield i, {
                    "question.idx": i,
                    "question.text": question,
                    "answer.target": target,
                    "answer.text": options,
                }
