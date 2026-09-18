import os
import pandas as pd
import numpy as np

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_FOLDER = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FOLDER = os.path.join(BASE_FOLDER, "data", "processed")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "data", "processed")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def find_column(df, text):
    """
    Find a column containing the given text.
    """
    for column in df.columns:
        if text.lower() in column.lower():
            return column

    return None


def numeric_value(df, column):
    """
    Convert a column to numeric safely.
    """
    if column is None:
        return pd.Series(np.nan, index=df.index)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# --------------------------------------------------
# LOAD A2
# --------------------------------------------------

a2_path = os.path.join(
    DATA_FOLDER,
    "RA2019_A2_cleaned.csv"
)

a2 = pd.read_csv(a2_path)

print("A2 loaded:", a2.shape)


# --------------------------------------------------
# REMOVE TOTAL ROW
# --------------------------------------------------

state_column = find_column(a2, "States/UTs")

if state_column:
    a2 = a2[
        a2[state_column].astype(str).str.strip().str.lower()
        != "total"
    ].copy()


# --------------------------------------------------
# IDENTIFY IMPORTANT A2 COLUMNS
# --------------------------------------------------

acc_2016 = find_column(
    a2,
    "Total Number of Road Accidents during 2016"
)

acc_2017 = find_column(
    a2,
    "Total Number of Road Accidents during 2017"
)

acc_2018 = find_column(
    a2,
    "Total Number of Road Accidents during 2018"
)

acc_2019 = find_column(
    a2,
    "Total Number of Road Accidents during 2019 - Numbers"
)


# --------------------------------------------------
# CREATE ML DATASET
# --------------------------------------------------

ml_data = pd.DataFrame()

ml_data["State_UT"] = a2[state_column].astype(str).str.strip()

ml_data["Accidents_2016"] = numeric_value(
    a2,
    acc_2016
)

ml_data["Accidents_2017"] = numeric_value(
    a2,
    acc_2017
)

ml_data["Accidents_2018"] = numeric_value(
    a2,
    acc_2018
)

ml_data["Accidents_2019"] = numeric_value(
    a2,
    acc_2019
)


# --------------------------------------------------
# CREATE HISTORICAL FEATURES
# --------------------------------------------------

ml_data["Average_Accidents_2016_2018"] = (
    ml_data[
        [
            "Accidents_2016",
            "Accidents_2017",
            "Accidents_2018"
        ]
    ].mean(axis=1)
)

ml_data["Accident_Change_2018_vs_2016"] = (
    ml_data["Accidents_2018"]
    - ml_data["Accidents_2016"]
)

ml_data["Accident_Growth_2018_vs_2016"] = (
    (
        ml_data["Accidents_2018"]
        - ml_data["Accidents_2016"]
    )
    /
    ml_data["Accidents_2016"].replace(0, np.nan)
) * 100


# --------------------------------------------------
# REMOVE INVALID RECORDS
# --------------------------------------------------

ml_data = ml_data.dropna(
    subset=[
        "Accidents_2016",
        "Accidents_2017",
        "Accidents_2018",
        "Accidents_2019"
    ]
)


# --------------------------------------------------
# SAVE DATASET
# --------------------------------------------------

output_path = os.path.join(
    OUTPUT_FOLDER,
    "ml_accident_dataset.csv"
)

ml_data.to_csv(
    output_path,
    index=False
)


print("\nML dataset created successfully!")
print("Rows:", ml_data.shape[0])
print("Columns:", ml_data.shape[1])
print("\nSaved to:")
print(output_path)

print("\nPreview:")
print(ml_data.head())