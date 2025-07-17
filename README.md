# Airbnb Price Prediction using Spark MLlib and MLflow

This project aims to build a price prediction model for Airbnb listings using distributed ML pipelines in PySpark. The goal is to compare multiple regression models and track experiments using MLflow.

---

Problem Statement

Build a regression system that predicts Airbnb listing prices using ML techniques, Spark DataFrames, and track all experiments using MLflow for reproducibility and performance comparison.

---

Objectives

- Load and preprocess Airbnb listing data.
- Train and evaluate multiple Spark MLlib models.
- Use Pipelines for clean workflow.
- Track each experiment in MLflow.
- Visualize and log feature importances for tree-based models.

---

Technologies Used

- Apache Spark (PySpark)
- Spark MLlib
- MLflow
- Pandas (for feature logging)

---

Dataset Info

- **Size:** 4.7k entries
- **Features:** 22 columns
- **Target:** `price`
- **Categorical:** `property_type`, `room_type`, `bed_type`

---

Preprocessing Steps

- Dropped irrelevant columns: `latitude`, `longitude`, `zipcode`, `neighbourhood_cleansed`
- Created interaction terms:
  - `accom_x_bedrooms = accommodates * bedrooms`
  - `reviews_x_rating = number_of_reviews * review_scores_rating`
- One-hot encoded categorical variables.
- Standardized continuous features.

---

Models Compared

| Model                | Spark Class                                    |
|---------------------|-------------------------------------------------|
| Linear Regression    | `LinearRegression`                             |
| Decision Tree        | `DecisionTreeRegressor`                        |
| Random Forest        | `RandomForestRegressor`                        |
| Gradient Boosted Tree| `GBTRegressor`                                 |

---

 Hyperparameter Tuning

Used `CrossValidator` with `ParamGridBuilder` to tune:
- `regParam`, `elasticNetParam` (for Linear Regression)
- `maxDepth`, `numTrees` (for Trees)
- `maxIter` (for GBT)

---

Evaluation Metrics

Each model is evaluated using:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score (Coefficient of Determination)**

---

 MLflow Tracking

- Tracked:
  - Model type
  - Hyperparameters
  - RMSE, MAE, R²
  - Feature importances (as CSV artifact)
- Set tracking URI: `http://localhost:5000`
- Visualized best runs via MLflow UI

---

 Sample CLI Usage

```bash
python airbnb_price_prediction.py --model rf
