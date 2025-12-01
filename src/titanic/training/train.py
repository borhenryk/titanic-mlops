# Databricks notebook source
# Titanic Survival Model Training with Hyperopt

import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from mlflow.models.signature import infer_signature

catalog = dbutils.widgets.get("catalog") if "catalog" in [w.name for w in dbutils.widgets.list()] else "dbdemos_henryk"
schema = dbutils.widgets.get("schema") if "schema" in [w.name for w in dbutils.widgets.list()] else "titanic_mlops"
experiment_name = dbutils.widgets.get("experiment_name") if "experiment_name" in [w.name for w in dbutils.widgets.list()] else "/Shared/titanic-mlops-dev"
model_name = dbutils.widgets.get("model_name") if "model_name" in [w.name for w in dbutils.widgets.list()] else "titanic_survival_model"

mlflow.set_experiment(experiment_name)

features_df = spark.table(f"{catalog}.{schema}.titanic_features")
pdf = features_df.toPandas()

feature_columns = ['Pclass', 'Sex_male', 'Age', 'Is_Child', 'Is_Teen', 'Is_Senior',
    'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Fare', 'Fare_log',
    'Embarked_C', 'Embarked_Q', 'Embarked_S', 'HasCabin',
    'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Master']

X = pdf[feature_columns]
y = pdf['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

search_space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200]),
    'max_depth': hp.choice('max_depth', [3, 5, 7, 10, None]),
    'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
}

def objective(params):
    with mlflow.start_run(nested=True):
        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", roc_auc)
        return {'loss': -roc_auc, 'status': STATUS_OK, 'model': model}

trials = Trials()
with mlflow.start_run(run_name="hyperopt_search"):
    best = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=15, trials=trials)

best_model = trials.results[np.argmin(trials.losses())]['model']
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

with mlflow.start_run(run_name="best_model"):
    mlflow.log_metrics({
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    })
    signature = infer_signature(X_train, best_model.predict(X_train))
    mlflow.sklearn.log_model(best_model, "model", signature=signature,
        registered_model_name=f"{catalog}.{schema}.{model_name}")

print(f"Model registered: {catalog}.{schema}.{model_name}")
