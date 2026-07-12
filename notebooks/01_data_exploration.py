import pandas as pd


# Column names from the UCI Cleveland Heart Disease dataset
columns = [
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


# Load dataset
data_path = "data/raw/heart_disease.csv"

df = pd.read_csv(
    data_path,
    names=columns,
    na_values="?"
)


print("=" * 50)
print("BIOINSIGHT AI — DATASET OVERVIEW")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Patients:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["target"].value_counts().sort_index())
print("\n" + "=" * 50)
print("DATASET VALIDATION")
print("=" * 50)

# Check number of rows and columns
assert df.shape == (303, 14), "Unexpected dataset shape!"

# Check for duplicate rows
duplicate_count = df.duplicated().sum()

print(f"\nDuplicate Rows: {duplicate_count}")

# Check target values
expected_targets = {0, 1, 2, 3, 4}
actual_targets = set(df["target"].unique())

assert actual_targets == expected_targets, "Unexpected target values!"

print("Dataset shape is correct.")
print("Target values are valid.")
print("Dataset validation completed successfully.")