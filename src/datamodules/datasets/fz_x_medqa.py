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


_DSET_URL = "https://drive.google.com/file/d/1pfK45VXSszEaV5HrVrB7gVJwOB1ofXtu/view?usp=sharing"
_DESCRIPTION = "A mapping between the FinzdZebra corpus and the MedQA dataset"
_VERSION = "0.0.1"
_HOMEPAGE = "https://github.com/MotzWanted/Open-Domain-MedQA"
_CITATION = ""


class FZxMedQADataset(datasets.GeneratorBasedBuilder):
    """ FZxMedQA Dataset. Version 0.0.1 """

    VERSION = datasets.Version(_VERSION)
    force = False

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "question_id": datasets.Value("int32"),
                    "answer_idx": datasets.Value("int32"),
                    "answer_choices": datasets.Sequence(datasets.Value("string")),
                    "document": datasets.Value("string"),
                    "is_gold": datasets.Value("bool"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION
        )

    @staticmethod
    def _get_drive_url(url):
        base_url = 'https://drive.google.com/uc?id='
        split_url = url.split('/')
        return base_url + split_url[5]

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(self._get_drive_url(_DSET_URL))

        return [datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "filepath": downloaded_file,
                "split": "all"
            },
        ), ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """
        with open(filepath, 'r') as f:
            for d in json.load(f)['data']:
                yield d['idx'], d
