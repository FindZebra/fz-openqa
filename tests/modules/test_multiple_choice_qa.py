from unittest import TestCase

import torch

from fz_openqa.modeling.__saved_models.multiple_choice_qa import MultipleChoiceQA


class TestMultipleChoiceQA(TestCase):
    # todo: re-implement this test
    def ____test_argmax_select(self):
        x01 = torch.randn((10,))
        x02 = torch.randn((10,))
        x11 = torch.randn((10,))
        x12 = torch.randn((10,))
        inputs = [
            {
                "a": torch.tensor([0, 0]),
                "b": torch.tensor([1.0, 0.0]),
                "v": torch.cat([x01[None], x02[None]]),
            },
            {
                "a": torch.tensor([1, 1]),
                "b": torch.tensor([0, 1.0]),
                "v": torch.cat([x11[None], x12[None]]),
            },
        ]
        output = MultipleChoiceQA.argmax_select(inputs=inputs, key="b")
        self.assertTrue(torch.all(torch.tensor([0, 1]) == output["a"]))
        self.assertTrue(torch.all(torch.cat([x01[None], x12[None]]) == output["v"]))
