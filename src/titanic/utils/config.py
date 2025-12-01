"""
Configuration utilities for Titanic MLOps pipeline.
"""

import os
from dataclasses import dataclass
import yaml


@dataclass
class Config:
    """Configuration class for Titanic MLOps pipeline."""
    catalog: str = "dbdemos_henryk"
    schema: str = "titanic_mlops"
    experiment_name: str = "/Shared/titanic-mlops"
    model_name: str = "titanic_survival_model"
    endpoint_name: str = "titanic-survival-endpoint"
    test_size: float = 0.2
    random_state: int = 42
    hyperopt_max_evals: int = 20
    min_accuracy: float = 0.75
    min_precision: float = 0.70
    min_recall: float = 0.65
    min_f1_score: float = 0.70
    min_roc_auc: float = 0.80
    
    @property
    def full_model_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.model_name}"
    
    def get_table_path(self, table_name: str) -> str:
        return f"{self.catalog}.{self.schema}.{table_name}"
    
    @property
    def acceptance_criteria(self) -> dict:
        return {
            'accuracy': self.min_accuracy,
            'precision': self.min_precision,
            'recall': self.min_recall,
            'f1_score': self.min_f1_score,
            'roc_auc': self.min_roc_auc
        }


FEATURE_COLUMNS = [
    'Pclass', 'Sex_male', 'Age', 'Is_Child', 'Is_Teen', 'Is_Senior',
    'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Fare', 'Fare_log',
    'Embarked_C', 'Embarked_Q', 'Embarked_S', 'HasCabin',
    'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Master'
]
