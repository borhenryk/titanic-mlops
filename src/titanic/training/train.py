# Databricks notebook source
# MAGIC %md
# MAGIC # Titanic Survival Model Training
# MAGIC Training pipeline with hyperparameter optimization using MLflow and Hyperopt

# COMMAND ----------

# DBTITLE 1,Configuration
import mlflow
from mlflow.tracking import MlflowClient

# Get parameters with try/except for serverless compatibility
try:
    catalog = dbutils.widgets.get("catalog")
except:
    catalog = "mcp_dabs_test"

try:
    schema = dbutils.widgets.get("schema")
except:
    schema = "titanic_mlops_dev"

try:
    experiment_name = dbutils.widgets.get("experiment_name")
except:
    experiment_name = "/Shared/titanic-mlops-dev"

try:
    model_name = dbutils.widgets.get("model_name")
except:
    model_name = "titanic_survival_model"

print(f"📦 Catalog: {catalog}")
print(f"📁 Schema: {schema}")
print(f"🧪 Experiment: {experiment_name}")
print(f"🤖 Model: {model_name}")

# COMMAND ----------

# DBTITLE 1,Setup MLflow Experiment
mlflow.set_experiment(experiment_name)
print(f"✅ MLflow experiment set: {experiment_name}")

# COMMAND ----------

# DBTITLE 1,Load Feature Data
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load features
features_df = spark.table(f"{catalog}.{schema}.titanic_features")
pdf = features_df.toPandas()

# Prepare features and target
feature_columns = [
    'Pclass', 'Sex_male', 'Age', 'Is_Child', 'Is_Teen', 'Is_Senior',
    'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Fare', 'Fare_log',
    'Embarked_C', 'Embarked_Q', 'Embarked_S', 'HasCabin',
    'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Master'
]

X = pdf[feature_columns]
y = pdf['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data loaded: {len(X_train)} train, {len(X_test)} test samples")

# COMMAND ----------

# DBTITLE 1,Hyperparameter Optimization
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

search_space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 150]),
    'max_depth': hp.choice('max_depth', [3, 5, 7, 10]),
    'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
}

def objective(params):
    with mlflow.start_run(nested=True):
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_proba)
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", roc_auc)
        
        return {'loss': -roc_auc, 'status': STATUS_OK, 'model': model}

print("🔍 Starting Hyperparameter Optimization...")
trials = Trials()

with mlflow.start_run(run_name="hyperopt_search") as parent_run:
    best = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=10, trials=trials)
    print(f"✅ Best ROC-AUC: {-min(trials.losses()):.4f}")

# COMMAND ----------

# DBTITLE 1,Final Model Metrics
best_trial_idx = np.argmin(trials.losses())
best_model = trials.results[best_trial_idx]['model']

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

final_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_proba)
}

print("📊 Final Model Metrics:")
for metric, value in final_metrics.items():
    print(f"   {metric}: {value:.4f}")

# COMMAND ----------

# DBTITLE 1,Register Model to Unity Catalog
from mlflow.models.signature import infer_signature

signature = infer_signature(X_train, best_model.predict(X_train))

with mlflow.start_run(run_name="best_model_registration") as run:
    mlflow.log_params(best_model.get_params())
    for metric, value in final_metrics.items():
        mlflow.log_metric(metric, value)
    
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name=f"{catalog}.{schema}.{model_name}"
    )
    
    print(f"✅ Model registered: {catalog}.{schema}.{model_name}")

# COMMAND ----------

# DBTITLE 1,Save Predictions
predictions_df = pdf[['PassengerId', 'label']].copy()
predictions_df['prediction'] = best_model.predict(X)
predictions_df['probability'] = best_model.predict_proba(X)[:, 1]

spark.createDataFrame(predictions_df).write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog}.{schema}.titanic_predictions"
)
print(f"✅ Predictions saved to {catalog}.{schema}.titanic_predictions")
