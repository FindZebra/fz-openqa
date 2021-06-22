from unittest import TestCase

import numpy as np
import torch

from fz_openqa.modeling.functional import padless_cat, count_trailing_padding, pad


class Test_functional(TestCase):

    def test_padless_cat_case(self):
        a = torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 0, 3, 0, 0]])
        b = torch.tensor([[4, 0, 5, 6], [4, 5, 0, 0]])
        ab = padless_cat(a, b, pad_token=0)
        self.assertTrue((1 - (ab == torch.tensor([[1, 2, 3, 4, 0, 5, 6], [1, 2, 0, 3, 4, 5, 0]])).float()).sum() == 0)

    def test_padless_cat_generated(self):
        length = 10
        bs = 3
        n_trials = 10

        def count(x: torch.Tensor, value):
            return [t for t in x.view(-1)].count(value)

        def gen_seqs(token, pad_token, length):
            l = np.random.randint(1, length - 1)
            return l * [token] + (length - l) * [pad_token]

        for k in range(n_trials):
            for pad_tok in [0, -100]:
                a = torch.tensor([gen_seqs(1, pad_tok, length) for _ in range(bs)])
                b = torch.tensor([gen_seqs(2, pad_tok, length) for _ in range(bs)])
                ab = padless_cat(a, b, pad_token=pad_tok)
                for xa, xb, xab in zip(a, b, ab):
                    self.assertEqual(count(xa, 1), count(xab, 1))
                    self.assertEqual(count(xb, 2), count(xab, 2))
                    self.assertLess(len(xab), len(xa) + len(xb))

    def test_pad(self):
        batch = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4, 0, 0])]
        padded = pad(batch, 0)
        self.assertEqual(padded.shape, torch.Size([2, 4]))

        batch = [torch.tensor([1, 2, 3, 0, 0]), torch.tensor([1, 2, 3, 4, 0, 0, 0, 0])]
        padded = pad(batch, 0)
        self.assertEqual(padded.shape, torch.Size([2, 4]))

        batch = [torch.tensor([1, 2, 3, 0, 0]), torch.tensor([0, 1, 2, 3, 4, 0, 0, 0, 0])]
        padded = pad(batch, 0)
        self.assertEqual(padded.shape, torch.Size([2, 5]))

    def test_count_trailing_padding(self):
        x = [1, 2, 3]
        self.assertEqual(count_trailing_padding(x, 0), 0)

        x = [1, 2, 3, 0, 0]
        print(count_trailing_padding(x, 0))
        self.assertEqual(count_trailing_padding(x, 0), 2)

        x = [1, 2, 0, 3, 'x', 'x']
        self.assertEqual(count_trailing_padding(x, 'x'), 2)

        x = [0, 1, 2, 0, 3]
        self.assertEqual(count_trailing_padding(x, 0), 0)
