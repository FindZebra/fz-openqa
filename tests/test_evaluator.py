from unittest import TestCase

from fz_openqa.modeling.models.base import Model


class TestEvaluator(TestCase):
    def test__filter_features_from_output(self):
        """test that only the key _feature_ is filtered out."""
        data = {'loss': None,
                '_feature_': None,
                '__flag__': None,
                '_miss': None,
                'miss_': None}
        filtered_data = Model._filter_features_from_output(data)
        self.assertEqual(set(filtered_data), set(data.keys()) - {'_feature_'})
