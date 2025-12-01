"""
Unit tests for configuration module.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from titanic.utils.config import Config, FEATURE_COLUMNS


class TestConfig:
    def test_default_config(self):
        config = Config()
        assert config.catalog == "dbdemos_henryk"
        assert config.schema == "titanic_mlops"
    
    def test_full_model_name(self):
        config = Config()
        expected = "dbdemos_henryk.titanic_mlops.titanic_survival_model"
        assert config.full_model_name == expected
    
    def test_acceptance_criteria(self):
        config = Config()
        criteria = config.acceptance_criteria
        assert 'accuracy' in criteria
        assert 'roc_auc' in criteria


class TestFeatureColumns:
    def test_feature_columns_not_empty(self):
        assert len(FEATURE_COLUMNS) > 0
    
    def test_expected_features_present(self):
        expected = ['Pclass', 'Sex_male', 'Age', 'Fare']
        for feature in expected:
            assert feature in FEATURE_COLUMNS
