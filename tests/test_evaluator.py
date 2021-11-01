from unittest import TestCase

from fz_openqa.modeling.modules.base import Module


class TestEvaluator(TestCase):
    def test__filter_features_from_output(self):
        """test that only the key _feature_ is filtered out."""
        data = {'loss': None,
                '_feature_': None,
                '__flag__': None,
                '_miss': None,
                'miss_': None}
        filtered_data = Module._filter_features_from_output(data)
        self.assertEqual(set(filtered_data), set(data.keys()) - {'_feature_'})
