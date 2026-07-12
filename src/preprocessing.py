from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "heart_disease.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_clean.csv"
)


# --------------------------------------------------
# DATASET COLUMNS
# --------------------------------------------------

COLUMNS = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "cholesterol",
    "fasting_blood_sugar",
    "resting_ecg",
    "max_heart_rate",
    "exercise_induced_angina",
    "st_depression",
    "st_slope",
    "major_vessels",
    "thalassemia",
    "target",
]


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_data():
    """Load the raw Cleveland Heart Disease dataset."""

    df = pd.read_csv(
        RAW_DATA_PATH,
        names=COLUMNS,
        na_values="?",
    )

    return df


# --------------------------------------------------
# CLEAN DATASET
# --------------------------------------------------

def clean_data(df):
    """Clean and preprocess the heart disease dataset."""

    df = df.copy()

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Count duplicate rows
    duplicate_count = df.duplicated().sum()

    print(f"\nDuplicate rows found: {duplicate_count}")

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert target into binary classification
    #
    # Original:
    # 0 = No Heart Disease
    # 1, 2, 3, 4 = Heart Disease Present

    df["target"] = (df["target"] > 0).astype(int)

    # Columns containing missing values
    missing_columns = [
        "major_vessels",
        "thalassemia",
    ]

    # Fill missing values using median
    for column in missing_columns:

        median_value = df[column].median()

        df[column] = df[column].fillna(median_value)

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    return df


# --------------------------------------------------
# VALIDATE CLEANED DATASET
# --------------------------------------------------

def validate_data(df):
    """Validate the processed dataset."""

    assert df.isnull().sum().sum() == 0, (
        "Dataset still contains missing values."
    )

    assert set(df["target"].unique()) == {0, 1}, (
        "Target column is not binary."
    )

    assert df.duplicated().sum() == 0, (
        "Dataset still contains duplicate rows."
    )

    print("\nDataset validation successful.")


# --------------------------------------------------
# SAVE CLEANED DATASET
# --------------------------------------------------

def save_data(df):
    """Save the cleaned dataset."""

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print(
        f"\nProcessed dataset saved to:\n"
        f"{PROCESSED_DATA_PATH}"
    )


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

def split_data(df):
    """Split processed data into training and testing datasets."""

    # Separate features and target
    X = df.drop(columns=["target"])

    y = df["target"]

    # Create training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\n" + "=" * 55)

    print("TRAIN / TEST SPLIT")

    print("=" * 55)

    print(f"\nX_train shape: {X_train.shape}")

    print(f"X_test shape: {X_test.shape}")

    print(f"y_train shape: {y_train.shape}")

    print(f"y_test shape: {y_test.shape}")

    print("\nTraining Target Distribution:")

    print(y_train.value_counts().sort_index())

    print("\nTesting Target Distribution:")

    print(y_test.value_counts().sort_index())

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    print("=" * 55)

    print("BIOINSIGHT AI — DATA PREPROCESSING PIPELINE")

    print("=" * 55)

    # Load raw dataset
    df = load_data()

    print(f"\nRaw dataset shape: {df.shape}")

    # Clean dataset
    df = clean_data(df)

    # Validate dataset
    validate_data(df)

    # Save processed dataset
    save_data(df)

    print(f"\nProcessed dataset shape: {df.shape}")

    print("\nTarget Distribution:")

    print(df["target"].value_counts().sort_index())

    # Split dataset for machine learning
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n" + "=" * 55)

    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")

    print("=" * 55)


# --------------------------------------------------
# RUN PIPELINE
# --------------------------------------------------

if __name__ == "__main__":
    main()