import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_FOLDER = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_FOLDER,
    "data",
    "processed",
    "ml_accident_dataset.csv"
)

MODEL_FOLDER = os.path.join(
    BASE_FOLDER,
    "models"
)

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

features = [
    "Accidents_2016",
    "Accidents_2017",
    "Accidents_2018",
    "Average_Accidents_2016_2018",
    "Accident_Change_2018_vs_2016",
    "Accident_Growth_2018_vs_2016"
]

target = "Accidents_2019"


# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------

model_data = df[
    features + [target]
].replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna()


X = model_data[features]
y = model_data[target]


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# MODELS
# --------------------------------------------------

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# --------------------------------------------------
# TRAIN AND EVALUATE
# --------------------------------------------------

results = []

trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²  :", round(r2, 4))


# --------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results_path = os.path.join(
    MODEL_FOLDER,
    "model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# --------------------------------------------------
# SELECT MODEL
# --------------------------------------------------

best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(),
    "Model"
]

best_model = trained_models[
    best_model_name
]


print("\nModel selected based on lowest RMSE:")
print(best_model_name)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model_path = os.path.join(
    MODEL_FOLDER,
    "accident_prediction_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)


# Save feature names
feature_path = os.path.join(
    MODEL_FOLDER,
    "feature_names.txt"
)

with open(
    feature_path,
    "w"
) as file:

    for feature in features:
        file.write(
            feature + "\n"
        )


print("\nModel saved successfully!")
print(model_path)