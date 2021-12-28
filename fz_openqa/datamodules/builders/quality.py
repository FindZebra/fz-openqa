import logging

import dill  # type: ignore

from . import MedQaBuilder
from fz_openqa.datamodules.generators import quality

logger = logging.getLogger(__name__)


class QuALITYBuilder(MedQaBuilder):
    # HuggingFace dataset id or local path to script
    dset_script_path_or_id = quality.__file__
    dset_name = "questions"


class ConcatMedQaBuilder(MedQaBuilder):
    """A MedQa dataset with concatenated questions and answers"""

    dset_script_path_or_id = quality.__file__
    dset_name = "documents"
