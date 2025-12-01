# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering for Titanic Survival Prediction
# MAGIC This notebook prepares features for model training

# COMMAND ----------

# DBTITLE 1,Configuration
import os

# Get parameters from job configuration or use defaults
try:
    catalog = dbutils.widgets.get("catalog")
except:
    catalog = "mcp_dabs_test"

try:
    schema = dbutils.widgets.get("schema")
except:
    schema = "titanic_mlops_dev"

print(f"📦 Catalog: {catalog}")
print(f"📁 Schema: {schema}")

# COMMAND ----------

# DBTITLE 1,Create Schema if not exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
print(f"✅ Schema {catalog}.{schema} ready")

# COMMAND ----------

# DBTITLE 1,Download and Load Titanic Data
import pandas as pd

# Download Titanic dataset
titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
pdf = pd.read_csv(titanic_url)
print(f"✅ Downloaded {len(pdf)} rows")

# Convert to Spark DataFrame and save as raw table
raw_df = spark.createDataFrame(pdf)
raw_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{schema}.titanic_raw")
print(f"✅ Raw data saved to {catalog}.{schema}.titanic_raw")

# COMMAND ----------

# DBTITLE 1,Feature Engineering
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

# Load raw data
raw_df = spark.table(f"{catalog}.{schema}.titanic_raw")

# Create engineered features
features_df = raw_df.select(
    # Target variable
    F.col("Survived").cast(IntegerType()).alias("label"),
    
    # Passenger class (already numeric)
    F.col("Pclass").cast(IntegerType()).alias("Pclass"),
    
    # Sex - encode as binary
    F.when(F.col("Sex") == "male", 1).otherwise(0).alias("Sex_male"),
    
    # Age - fill nulls with median, create age groups
    F.coalesce(F.col("Age"), F.lit(28.0)).alias("Age"),
    F.when(F.col("Age") < 12, 1).otherwise(0).alias("Is_Child"),
    F.when((F.col("Age") >= 12) & (F.col("Age") < 18), 1).otherwise(0).alias("Is_Teen"),
    F.when(F.col("Age") >= 60, 1).otherwise(0).alias("Is_Senior"),
    
    # Family features
    F.col("SibSp").cast(IntegerType()),
    F.col("Parch").cast(IntegerType()),
    (F.col("SibSp") + F.col("Parch") + 1).alias("FamilySize"),
    F.when(F.col("SibSp") + F.col("Parch") == 0, 1).otherwise(0).alias("IsAlone"),
    
    # Fare - fill nulls with median
    F.coalesce(F.col("Fare"), F.lit(14.45)).alias("Fare"),
    F.log1p(F.coalesce(F.col("Fare"), F.lit(14.45))).alias("Fare_log"),
    
    # Embarked - one-hot encode
    F.when(F.col("Embarked") == "C", 1).otherwise(0).alias("Embarked_C"),
    F.when(F.col("Embarked") == "Q", 1).otherwise(0).alias("Embarked_Q"),
    F.when(F.col("Embarked") == "S", 1).otherwise(0).alias("Embarked_S"),
    
    # Title extraction from Name
    F.when(F.col("Name").contains("Mr."), "Mr")
     .when(F.col("Name").contains("Mrs."), "Mrs")
     .when(F.col("Name").contains("Miss."), "Miss")
     .when(F.col("Name").contains("Master."), "Master")
     .otherwise("Other").alias("Title"),
    
    # Cabin - has cabin indicator
    F.when(F.col("Cabin").isNotNull(), 1).otherwise(0).alias("HasCabin"),
    
    # Original ID for tracking
    F.col("PassengerId")
)

# Add title encoding
features_df = features_df.withColumn(
    "Title_Mr", F.when(F.col("Title") == "Mr", 1).otherwise(0)
).withColumn(
    "Title_Mrs", F.when(F.col("Title") == "Mrs", 1).otherwise(0)
).withColumn(
    "Title_Miss", F.when(F.col("Title") == "Miss", 1).otherwise(0)
).withColumn(
    "Title_Master", F.when(F.col("Title") == "Master", 1).otherwise(0)
).drop("Title")

print(f"✅ Created {len(features_df.columns)} features")
features_df.printSchema()

# COMMAND ----------

# DBTITLE 1,Save Feature Table
# Save as Delta table
features_df.write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog}.{schema}.titanic_features"
)

print(f"✅ Features saved to {catalog}.{schema}.titanic_features")
print(f"📊 Total records: {features_df.count()}")

# COMMAND ----------

# DBTITLE 1,Display Feature Statistics
display(features_df.describe())

# COMMAND ----------

# DBTITLE 1,Feature Correlation Analysis
import pandas as pd

# Convert to pandas for correlation
pdf = features_df.drop("PassengerId").toPandas()
correlation = pdf.corr()['label'].sort_values(ascending=False)
print("📈 Feature Correlation with Survival:")
print(correlation.to_string())
