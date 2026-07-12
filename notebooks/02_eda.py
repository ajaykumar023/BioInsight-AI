from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_clean.csv"
)

FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"

FIGURES_PATH.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)


print("=" * 60)
print("BIOINSIGHT AI — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nDataset Shape: {df.shape}")

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe().round(2))

print("\nTarget Distribution:")
print(df["target"].value_counts())

print("\nTarget Percentage:")
print(
    df["target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def save_figure(filename):
    """Save the current Matplotlib figure."""

    output_path = FIGURES_PATH / filename

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


# --------------------------------------------------
# FIGURE 1 — TARGET DISTRIBUTION
# --------------------------------------------------

target_counts = df["target"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    ["No Heart Disease", "Heart Disease"],
    target_counts.values,
)

plt.title("Heart Disease Target Distribution")

plt.xlabel("Diagnosis")

plt.ylabel("Number of Patients")

save_figure("01_target_distribution.png")


# --------------------------------------------------
# FIGURE 2 — AGE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["age"],
    bins=15,
    edgecolor="black",
)

plt.title("Age Distribution of Patients")

plt.xlabel("Age")

plt.ylabel("Number of Patients")

save_figure("02_age_distribution.png")


# --------------------------------------------------
# FIGURE 3 — HEART DISEASE BY SEX
# --------------------------------------------------

sex_disease_table = pd.crosstab(
    df["sex"],
    df["target"],
)

sex_disease_table.index = ["Female", "Male"]

sex_disease_table.columns = [
    "No Heart Disease",
    "Heart Disease",
]

sex_disease_table.plot(
    kind="bar",
    figsize=(8, 5),
)

plt.title("Heart Disease Distribution by Sex")

plt.xlabel("Sex")

plt.ylabel("Number of Patients")

plt.xticks(rotation=0)

save_figure("03_heart_disease_by_sex.png")


# --------------------------------------------------
# FIGURE 4 — HEART DISEASE BY CHEST PAIN TYPE
# --------------------------------------------------

chest_pain_table = pd.crosstab(
    df["chest_pain_type"],
    df["target"],
)

chest_pain_table.columns = [
    "No Heart Disease",
    "Heart Disease",
]

chest_pain_table.plot(
    kind="bar",
    figsize=(9, 5),
)

plt.title("Heart Disease Distribution by Chest Pain Type")

plt.xlabel("Chest Pain Type")

plt.ylabel("Number of Patients")

plt.xticks(rotation=0)

save_figure("04_heart_disease_by_chest_pain.png")


# --------------------------------------------------
# FIGURE 5 — CORRELATION HEATMAP
# --------------------------------------------------

correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(13, 10))

heatmap = plt.imshow(
    correlation_matrix,
    aspect="auto",
    cmap="coolwarm",
)

plt.colorbar(heatmap)

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90,
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
)

plt.title("Feature Correlation Matrix")

save_figure("05_correlation_heatmap.png")


# --------------------------------------------------
# FIGURE 6 — CLINICAL FEATURE COMPARISON
# --------------------------------------------------

clinical_features = [
    "age",
    "resting_blood_pressure",
    "cholesterol",
    "max_heart_rate",
    "st_depression",
]

clinical_means = (
    df.groupby("target")[clinical_features]
    .mean()
    .T
)

clinical_means.columns = [
    "No Heart Disease",
    "Heart Disease",
]

clinical_means.plot(
    kind="bar",
    figsize=(11, 6),
)

plt.title("Average Clinical Features by Diagnosis")

plt.xlabel("Clinical Feature")

plt.ylabel("Average Value")

plt.xticks(rotation=30)

save_figure("06_clinical_feature_comparison.png")


# --------------------------------------------------
# COMPLETION MESSAGE
# --------------------------------------------------

print("\n" + "=" * 60)

print("EDA COMPLETED SUCCESSFULLY")

print("=" * 60)

print(f"\nFigures saved inside:\n{FIGURES_PATH}")