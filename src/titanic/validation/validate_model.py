# Databricks notebook source
# Model Validation

import mlflow
from mlflow import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

catalog = dbutils.widgets.get("catalog") if "catalog" in [w.name for w in dbutils.widgets.list()] else "dbdemos_henryk"
schema = dbutils.widgets.get("schema") if "schema" in [w.name for w in dbutils.widgets.list()] else "titanic_mlops"
model_name = dbutils.widgets.get("model_name") if "model_name" in [w.name for w in dbutils.widgets.list()] else "titanic_survival_model"

full_model_name = f"{catalog}.{schema}.{model_name}"
client = MlflowClient()

versions = client.search_model_versions(f"name='{full_model_name}'")
latest = max(versions, key=lambda x: int(x.version))
model = mlflow.sklearn.load_model(f"models:/{full_model_name}/{latest.version}")

features_df = spark.table(f"{catalog}.{schema}.titanic_features")
pdf = features_df.toPandas()

feature_columns = ['Pclass', 'Sex_male', 'Age', 'Is_Child', 'Is_Teen', 'Is_Senior',
    'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Fare', 'Fare_log',
    'Embarked_C', 'Embarked_Q', 'Embarked_S', 'HasCabin',
    'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Master']

X = pdf[feature_columns]
y = pdf['label']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_proba)
}

thresholds = {'accuracy': 0.75, 'precision': 0.70, 'recall': 0.65, 'f1_score': 0.70, 'roc_auc': 0.80}
all_passed = all(metrics[k] >= v for k, v in thresholds.items())

if all_passed:
    client.set_registered_model_alias(name=full_model_name, alias="Champion", version=latest.version)
    print(f"Model {latest.version} promoted to Champion")
else:
    raise ValueError("Validation failed")
