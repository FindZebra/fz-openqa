from fz_openqa.datamodules.pipes import Gate
from fz_openqa.datamodules.pipes import RenameKeys
from fz_openqa.datamodules.pipes import SearchCorpus
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.utils.condition import HasKeyWithPrefix
from fz_openqa.datamodules.utils.condition import Not
from fz_openqa.datamodules.utils.condition import Reduce
from fz_openqa.datamodules.utils.condition import Static


class MaybeSearchDocuments(Gate):
    def __init__(self, n_documents: int, *, corpus):
        # Search corpus pipe: this pipe is activated only if
        # the batch does not already contains documents (Gate).
        # todo: do not initialize SearchCorpus when no corpus is available
        activate_doc_search = Reduce(
            Static(n_documents > 0),
            Not(HasKeyWithPrefix("document.")),
            reduce_op=all,
        )
        super(MaybeSearchDocuments, self).__init__(
            activate_doc_search,
            Sequential(SearchCorpus(corpus, k=n_documents)),
        )
