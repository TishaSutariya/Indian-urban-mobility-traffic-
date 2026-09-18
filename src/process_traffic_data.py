import pandas as pd
import os
import glob


# ============================================================
# 1. FOLDERS
# ============================================================

DATA_FOLDER = "data"
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "processed")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. READ CSV SAFELY
# ============================================================

def read_csv_safely(file_path):

    encodings = ["utf-8", "latin1", "cp1252"]

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read file: {file_path}")


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    return df


# ============================================================
# 4. CLEAN TEXT VALUES
# ============================================================

def clean_text_values(df):

    for column in df.select_dtypes(include="object").columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .replace({
                "nan": pd.NA,
                "NaN": pd.NA,
                "N/A": pd.NA,
                "NA": pd.NA,
                "-": pd.NA,
                "--": pd.NA
            })
        )

    return df


# ============================================================
# 5. CONVERT NUMERIC COLUMNS
# ============================================================

def convert_numeric_columns(df):

    for column in df.columns:

        if df[column].dtype == "object":

            converted = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            # Convert only when most values can actually
            # be interpreted as numbers.
            valid_ratio = converted.notna().mean()

            if valid_ratio >= 0.70:
                df[column] = converted

    return df


# ============================================================
# 6. REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

def remove_empty_columns(df):

    df = df.dropna(axis=1, how="all")

    return df


# ============================================================
# 7. REMOVE COMPLETELY EMPTY ROWS
# ============================================================

def remove_empty_rows(df):

    df = df.dropna(axis=0, how="all")

    return df


# ============================================================
# 8. PROCESS ONE FILE
# ============================================================

def process_file(file_path):

    file_name = os.path.basename(file_path)

    print("\n" + "=" * 70)
    print("Processing:", file_name)

    # Read
    df = read_csv_safely(file_path)

    original_rows = len(df)
    original_columns = len(df.columns)

    print("Original rows:", original_rows)
    print("Original columns:", original_columns)

    # Clean
    df = clean_column_names(df)
    df = clean_text_values(df)
    df = convert_numeric_columns(df)
    df = remove_empty_columns(df)
    df = remove_empty_rows(df)

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates()

    # Output file name
    output_name = file_name.replace(
        ".csv",
        "_cleaned.csv"
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_name
    )

    # Save
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print("Final rows:", len(df))
    print("Final columns:", len(df.columns))
    print("Duplicates removed:", duplicate_count)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Saved to:", output_path)

    return {
        "dataset": file_name,
        "original_rows": original_rows,
        "cleaned_rows": len(df),
        "original_columns": original_columns,
        "cleaned_columns": len(df.columns),
        "duplicates_removed": duplicate_count,
        "missing_values": int(df.isna().sum().sum())
    }


# ============================================================
# 9. PROCESS ALL RA2019 CSV FILES
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(DATA_FOLDER, "RA2019_A*.csv")
    )
)

if not files:

    print("No RA2019 CSV files found.")

else:

    summary = []

    for file_path in files:

        try:

            result = process_file(file_path)

            summary.append(result)

        except Exception as error:

            print(
                f"ERROR while processing "
                f"{os.path.basename(file_path)}:"
            )

            print(error)


# ============================================================
# 10. SAVE DATASET SUMMARY
# ============================================================

if summary:

    summary_df = pd.DataFrame(summary)

    summary_path = os.path.join(
        OUTPUT_FOLDER,
        "dataset_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("ALL DATASETS PROCESSED SUCCESSFULLY")
    print("=" * 70)

    print(summary_df.to_string(index=False))

    print("\nSummary saved to:")
    print(summary_path)

else:

    print("\nNo datasets were processed.")