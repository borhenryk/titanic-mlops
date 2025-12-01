# 🚢 Titanic MLOps Pipeline

A production-ready ML pipeline for Titanic survival prediction, built with **Databricks Asset Bundles** and following MLOps best practices.

## 🎯 Project Overview

This project demonstrates an end-to-end MLOps pipeline that:
- Trains a machine learning model on the classic Titanic dataset
- Uses **hyperparameter optimization** with Hyperopt
- Tracks experiments with **MLflow**
- Registers models to **Unity Catalog**
- Deploys models as **Serving Endpoints**
- Implements **CI/CD** with GitHub Actions

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 82.68% |
| Precision | 79.69% |
| Recall | 73.91% |
| F1 Score | 76.69% |
| ROC AUC | 84.98% |

## 🚀 Quick Start

```bash
# Validate the bundle
databricks bundle validate -t dev

# Deploy to dev environment
databricks bundle deploy -t dev

# Run the training job
databricks bundle run -t dev titanic_training_job
```

## 📁 Project Structure

```
titanic-mlops/
├── databricks.yml              # DABs bundle configuration
├── resources/                  # Databricks resource definitions
├── src/titanic/               # Python source code
├── config/                    # Environment configs
├── scripts/                   # Deployment scripts
├── tests/                     # Unit tests
└── .github/workflows/         # CI/CD pipelines
```

## 🔗 Links

- **MLflow Experiment:** https://e2-demo-field-eng.cloud.databricks.com/ml/experiments/567797472287066
- **Model Registry:** dbdemos_henryk.titanic_mlops.titanic_survival_model
- **Serving Endpoint:** titanic-survival-endpoint-dev
- **CI/CD Workflows:** https://github.com/borhenryk/titanic-mlops/actions

## 📄 License

MIT License
