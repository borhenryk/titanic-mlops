# Databricks notebook source
# Feature Engineering for Titanic Survival Prediction

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

catalog = dbutils.widgets.get("catalog") if "catalog" in [w.name for w in dbutils.widgets.list()] else "dbdemos_henryk"
schema = dbutils.widgets.get("schema") if "schema" in [w.name for w in dbutils.widgets.list()] else "titanic_mlops"

raw_df = spark.table(f"{catalog}.{schema}.titanic_raw")

features_df = raw_df.select(
    F.col("Survived").cast(IntegerType()).alias("label"),
    F.col("Pclass").cast(IntegerType()).alias("Pclass"),
    F.when(F.col("Sex") == "male", 1).otherwise(0).alias("Sex_male"),
    F.coalesce(F.col("Age"), F.lit(28.0)).alias("Age"),
    F.when(F.col("Age") < 12, 1).otherwise(0).alias("Is_Child"),
    F.when((F.col("Age") >= 12) & (F.col("Age") < 18), 1).otherwise(0).alias("Is_Teen"),
    F.when(F.col("Age") >= 60, 1).otherwise(0).alias("Is_Senior"),
    F.col("SibSp").cast(IntegerType()),
    F.col("Parch").cast(IntegerType()),
    (F.col("SibSp") + F.col("Parch") + 1).alias("FamilySize"),
    F.when(F.col("SibSp") + F.col("Parch") == 0, 1).otherwise(0).alias("IsAlone"),
    F.coalesce(F.col("Fare"), F.lit(14.45)).alias("Fare"),
    F.log1p(F.coalesce(F.col("Fare"), F.lit(14.45))).alias("Fare_log"),
    F.when(F.col("Embarked") == "C", 1).otherwise(0).alias("Embarked_C"),
    F.when(F.col("Embarked") == "Q", 1).otherwise(0).alias("Embarked_Q"),
    F.when(F.col("Embarked") == "S", 1).otherwise(0).alias("Embarked_S"),
    F.when(F.col("Name").contains("Mr."), 1).otherwise(0).alias("Title_Mr"),
    F.when(F.col("Name").contains("Mrs."), 1).otherwise(0).alias("Title_Mrs"),
    F.when(F.col("Name").contains("Miss."), 1).otherwise(0).alias("Title_Miss"),
    F.when(F.col("Name").contains("Master."), 1).otherwise(0).alias("Title_Master"),
    F.when(F.col("Cabin").isNotNull(), 1).otherwise(0).alias("HasCabin"),
    F.col("PassengerId")
)

features_df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{schema}.titanic_features")
print(f"Features saved to {catalog}.{schema}.titanic_features")
