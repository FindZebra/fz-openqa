from copy import deepcopy
from unittest import TestCase

import dill

from fz_openqa.datamodules.pipes import Gate, Identity


class TestGate(TestCase):
    def setUp(self):
        self._batch = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        self.conds = [True, False, lambda x: True, lambda x: False]
        self.outputs = [self._batch, {}, self._batch, {}]

    @property
    def batch(self):
        return deepcopy(self._batch)

    def test_output_keys(self):
        for update in [False, True]:
            for cond in self.conds:
                for alt in [Identity(), None]:
                    pipe = Gate(cond, pipe=Identity(), alt=alt, update=update)
                    output = pipe(self.batch)
                    self.assertEqual(set(output.keys()),
                                     set(pipe.output_keys(list(self.batch.keys()))))

    def test__call(self):
        for update in [False, True]:
            for cond, _exp_output in zip(self.conds, self.outputs):
                for alt in [None, lambda x: {'w': None}]:

                    # build pipe and process batch
                    pipe = Gate(cond, pipe=Identity(), alt=alt, update=update)
                    output = pipe(self.batch)

                    # transform the expected output based on arguments
                    expected_output = deepcopy(_exp_output)
                    if alt is not None:
                        if len(expected_output) == 0:
                            expected_output = {'w': None}
                    if update:
                        expected_output = {**expected_output, **self.batch}

                    # compare output to expected output
                    self.assertEqual(set(expected_output.keys()), set(output.keys()))

    def test_pickle(self):
        for update in [False, True]:
            for cond in self.conds:
                for alt in [None, lambda x: {'w': None}]:
                    pipe = Gate(cond, pipe=Identity(), alt=alt, update=update)
                    self.assertTrue(dill.pickles(pipe))
