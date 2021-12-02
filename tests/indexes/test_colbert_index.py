import os
import shutil
import tempfile
import unittest

from datasets import Dataset
from pytorch_lightning import Trainer, LightningModule
from transformers import BertPreTrainedModel, AutoModel

from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.index.colbert import ColbertIndex
from fz_openqa.modeling.zero_shot import ZeroShot
from fz_openqa.utils.datastruct import Batch
from tests.indexes.test_base_index import TestIndex
from tests.indexes.test_dense_index import TestFaissIndex


class TestColbertIndex(TestFaissIndex):
    """Test the faiss colbert index without using the Trainer to process data with the model"""
    cls: Index.__class__ = ColbertIndex
