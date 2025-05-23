import argparse
import pandas as pd
import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Argument parsing (optional if running in Jupyter or IDE)
def parse_arguments():
    parser = argparse.ArgumentParser(description='Airbnb Price Prediction')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'lr', 'dt', 'rf', 'gbt'],
                        help='Model to train')
    return parser.parse_args()

# Spark session
spark = SparkSession.builder.appName("airbnb").getOrCreate()

# Load and inspect data
df = spark.read.csv('airbnb.csv', header=True, inferSchema=True)
df.printSchema()
df.show(5)

# Check for nulls
na_counts = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
na_counts.show()

# Drop unneeded columns
df = df.drop("latitude", "longitude", "zipcode", "neighbourhood_cleansed")

# Create interaction variables
df = df.withColumn("accom_x_bedrooms", col("accommodates") * col("bedrooms"))
df = df.withColumn("reviews_x_rating", col("number_of_reviews") * col("review_scores_rating"))

# Check duplicates
duplicate_count = df.count() - df.dropDuplicates().count()
print(f"Duplicate rows: {duplicate_count}")

# Preprocessing
categorical_columns = ["property_type", "room_type", "bed_type"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_indexed") for col in categorical_columns]
encoders = [OneHotEncoder(inputCol=col + "_indexed", outputCol=col + "_encoded") for col in categorical_columns]

feature_cols = [
    "host_total_listings_count", "accommodates", "bathrooms", "bedrooms",
    "beds", "minimum_nights", "number_of_reviews", "review_scores_rating",
    "review_scores_accuracy", "review_scores_cleanliness", "review_scores_value",
    "accom_x_bedrooms", "reviews_x_rating"
] + [c + "_encoded" for c in categorical_columns]



assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")
scaler = StandardScaler(inputCol="assembled_features", outputCol="features")

pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
pipeline_model = pipeline.fit(df)
df = pipeline_model.transform(df)


df.select("features", "price").show(5)

# Train/test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Define models
models = {
    "LinearRegression": LinearRegression(featuresCol="features", labelCol="price"),
    "DecisionTreeRegressor": DecisionTreeRegressor(featuresCol="features", labelCol="price"),
    "RandomForestRegressor": RandomForestRegressor(featuresCol="features", labelCol="price"),
    "GBTRegressor": GBTRegressor(featuresCol="features", labelCol="price")
}

evaluator = RegressionEvaluator(metricName="rmse", labelCol="price")

# Track best model
best_model_name = None
best_rmse = float('inf')
best_r2 = float('-inf')
best_mae = float('inf')
best_pipeline_model = None

# Train and evaluate models
for model_name, model in models.items():
    print(f"🔵 Training {model_name}...")

    paramGrid = ParamGridBuilder()
    if model_name == "LinearRegression":
        paramGrid = paramGrid.addGrid(model.regParam, [0.01, 0.1]) \
                             .addGrid(model.elasticNetParam, [0.0, 0.5, 1.0])
    elif model_name == "DecisionTreeRegressor":
        paramGrid = paramGrid.addGrid(model.maxDepth, [5, 10])
    elif model_name == "RandomForestRegressor":
        paramGrid = paramGrid.addGrid(model.numTrees, [20, 50]) \
                             .addGrid(model.maxDepth, [5, 10])
    elif model_name == "GBTRegressor":
        paramGrid = paramGrid.addGrid(model.maxIter, [25, 50]) \
                             .addGrid(model.maxDepth, [5, 10])

    paramGrid = paramGrid.build()

    cv = CrossValidator(
        estimator=model,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=5,
        parallelism=1
    )

    cv_model = cv.fit(train_data)

    predictions = cv_model.transform(test_data)
    rmse = evaluator.evaluate(predictions)
    r2 = RegressionEvaluator(metricName="r2", labelCol="price").evaluate(predictions)
    mae = RegressionEvaluator(metricName="mae", labelCol="price").evaluate(predictions)

    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_r2 = r2
        best_mae = mae
        best_model_name = model_name
        best_pipeline_model = cv_model.bestModel

# MLflow tracking
mlflow.set_experiment("AirbnbPricePrediction")

with mlflow.start_run(run_name=f"{best_model_name}_crossval"):
    mlflow.log_param("model_name", best_model_name)
    mlflow.log_metric("rmse", best_rmse)
    mlflow.log_metric("r2", best_r2)
    mlflow.log_metric("mae", best_mae)

    if hasattr(best_pipeline_model, "featureImportances"):
        importances = best_pipeline_model.featureImportances
        features = assembler.getInputCols()
        pandasDF = pd.DataFrame(
            list(zip(features, importances)),
            columns=["feature", "importance"]
        ).sort_values(by="importance", ascending=False)

        pandasDF.to_csv("feature-importance.csv", index=False)
        mlflow.log_artifact("feature-importance.csv")

    mlflow.spark.log_model(best_pipeline_model, f"{best_model_name}_best_model")

    print(f"✅ {best_model_name} logged with RMSE={best_rmse:.3f}")

print(f"\n🏆 Best Model: {best_model_name} with RMSE: {best_rmse:.3f}")
