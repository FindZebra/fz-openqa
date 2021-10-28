import datasets


class CorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for the Corpus objecxt."""

    def __init__(self, **kwargs):
        """BuilderConfig for the Corpus object.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CorpusConfig, self).__init__(**kwargs)


_DESCRIPTION = "A generic class to handle a large corpus of documents"
_VERSION = "0.0.1"
_HOMEPAGE = "https://github.com/MotzWanted/Open-Domain-MedQA"
_CITATION = ""


class CorpusDataset(datasets.GeneratorBasedBuilder):
    """CorpusDataset Dataset. Version 0.0.1"""

    VERSION = datasets.Version(_VERSION)
    force = False

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "idx": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "title": datasets.Value("string"),
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

    def _generate_examples(self):
        """Yields examples."""
        for i, fn in enumerate(self.config.data_files):
            with open(fn, "r") as f:
                yield i, {"idx": i, "text": f.read(), "title": "title"}
