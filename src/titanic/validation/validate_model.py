# Databricks notebook source
# MAGIC %md
# MAGIC # Model Validation
# MAGIC Validates the trained model against acceptance criteria before deployment

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
    model_name = dbutils.widgets.get("model_name")
except:
    model_name = "titanic_survival_model"

full_model_name = f"{catalog}.{schema}.{model_name}"
print(f"🔍 Validating model: {full_model_name}")

# COMMAND ----------

# DBTITLE 1,Define Acceptance Criteria
ACCEPTANCE_CRITERIA = {
    'accuracy': 0.70,
    'precision': 0.65,
    'recall': 0.60,
    'f1_score': 0.65,
    'roc_auc': 0.75
}

print("📋 Acceptance Criteria:")
for metric, threshold in ACCEPTANCE_CRITERIA.items():
    print(f"   {metric}: >= {threshold:.2f}")

# COMMAND ----------

# DBTITLE 1,Load Latest Model Version
client = MlflowClient()

model_versions = client.search_model_versions(f"name='{full_model_name}'")
if not model_versions:
    raise ValueError(f"No model versions found for {full_model_name}")

latest_version = max(model_versions, key=lambda x: int(x.version))
print(f"\n✅ Found model version: {latest_version.version}")

model_uri = f"models:/{full_model_name}/{latest_version.version}"
model = mlflow.sklearn.load_model(model_uri)
print(f"✅ Model loaded from: {model_uri}")

# COMMAND ----------

# DBTITLE 1,Load Test Data and Validate
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

features_df = spark.table(f"{catalog}.{schema}.titanic_features")
pdf = features_df.toPandas()

feature_columns = [
    'Pclass', 'Sex_male', 'Age', 'Is_Child', 'Is_Teen', 'Is_Senior',
    'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Fare', 'Fare_log',
    'Embarked_C', 'Embarked_Q', 'Embarked_S', 'HasCabin',
    'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Master'
]

X = pdf[feature_columns]
y = pdf['label']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

validation_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_proba)
}

print("\n📊 Validation Metrics:")
for metric, value in validation_metrics.items():
    print(f"   {metric}: {value:.4f}")

# COMMAND ----------

# DBTITLE 1,Check Acceptance Criteria
all_passed = True

print("\n" + "=" * 50)
print("🔍 VALIDATION RESULTS")
print("=" * 50)

for metric, threshold in ACCEPTANCE_CRITERIA.items():
    actual_value = validation_metrics[metric]
    passed = actual_value >= threshold
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {metric:12s}: {actual_value:.4f} (threshold: {threshold:.2f})")
    if not passed:
        all_passed = False

print("=" * 50)

if all_passed:
    print("\n🎉 MODEL VALIDATION PASSED!")
    client.set_registered_model_alias(name=full_model_name, alias="Champion", version=latest_version.version)
    print(f"✅ Model version {latest_version.version} promoted to 'Champion'")
else:
    print("\n⚠️ MODEL VALIDATION FAILED!")
    raise ValueError("Model validation failed - does not meet acceptance criteria")
