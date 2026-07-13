from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model.joblib"
)

MISCLASSIFIED_PATH = (
    PROJECT_ROOT
    / "reports"
    / "misclassified_patients.csv"
)

FIGURES_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
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
# LOAD DATA
# --------------------------------------------------

def load_data():
    """Load and prepare the raw dataset exactly as in training."""

    df = pd.read_csv(
        RAW_DATA_PATH,
        names=COLUMNS,
        na_values="?",
    )

    df = df.drop_duplicates().reset_index(drop=True)

    df["target"] = (
        df["target"] > 0
    ).astype(int)

    return df


# --------------------------------------------------
# RECREATE TEST SET
# --------------------------------------------------

def create_test_set(df):
    """Recreate the exact held-out test split used during training."""

    X = df.drop(columns=["target"])
    y = df["target"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_test, y_test


# --------------------------------------------------
# LOAD SAVED MODEL
# --------------------------------------------------

def load_model():
    """Load the final trained ML pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found:\n{MODEL_PATH}\n"
            "Run python src\\train.py first."
        )

    model = joblib.load(MODEL_PATH)

    return model


# --------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """Calculate final metrics and predictions."""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return metrics, predictions, probabilities


# --------------------------------------------------
# PRINT METRICS
# --------------------------------------------------

def print_metrics(metrics):
    """Print final evaluation metrics."""

    print("\n" + "=" * 65)

    print("SAVED MODEL METRICS")

    print("=" * 65)

    print(f"\nAccuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

def print_classification_report(y_test, predictions):
    """Print precision, recall, and F1 for each class."""

    print("\n" + "=" * 65)

    print("CLASSIFICATION REPORT")

    print("=" * 65)

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "No Heart Disease",
            "Heart Disease",
        ],
        digits=4,
        zero_division=0,
    )

    print("\n" + report)


# --------------------------------------------------
# CONFUSION MATRIX ANALYSIS
# --------------------------------------------------

def analyze_confusion_matrix(y_test, predictions):
    """Calculate and print TN, FP, FN, and TP."""

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    tn, fp, fn, tp = matrix.ravel()

    print("\n" + "=" * 65)

    print("ERROR ANALYSIS")

    print("=" * 65)

    print(f"\nTrue Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")

    print(
        "\nFalse Negatives are disease-positive patients "
        "that the model incorrectly predicted as no disease."
    )

    print(
        "False Positives are disease-negative patients "
        "that the model incorrectly predicted as disease."
    )

    return tn, fp, fn, tp


# --------------------------------------------------
# SAVE MISCLASSIFIED PATIENTS
# --------------------------------------------------

def save_misclassified_patients(
    X_test,
    y_test,
    predictions,
    probabilities,
):
    """Save test rows predicted incorrectly."""

    analysis_df = X_test.copy()

    analysis_df["actual_target"] = y_test

    analysis_df["predicted_target"] = predictions

    analysis_df["predicted_probability"] = probabilities

    analysis_df["error_type"] = "Correct"

    false_positive_mask = (
        (analysis_df["actual_target"] == 0)
        & (analysis_df["predicted_target"] == 1)
    )

    false_negative_mask = (
        (analysis_df["actual_target"] == 1)
        & (analysis_df["predicted_target"] == 0)
    )

    analysis_df.loc[
        false_positive_mask,
        "error_type",
    ] = "False Positive"

    analysis_df.loc[
        false_negative_mask,
        "error_type",
    ] = "False Negative"

    misclassified_df = analysis_df[
        analysis_df["error_type"] != "Correct"
    ].copy()

    MISCLASSIFIED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    misclassified_df.to_csv(
        MISCLASSIFIED_PATH,
        index=True,
        index_label="original_patient_index",
    )

    print(
        f"\nMisclassified patient records saved to:\n"
        f"{MISCLASSIFIED_PATH}"
    )

    print(
        f"\nTotal Misclassified Patients: "
        f"{len(misclassified_df)}"
    )

    return misclassified_df


# --------------------------------------------------
# ERROR ANALYSIS FIGURE
# --------------------------------------------------

def save_error_analysis_figure(tn, fp, fn, tp):
    """Save a bar chart of prediction outcomes."""

    FIGURES_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = [
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive",
    ]

    values = [
        tn,
        fp,
        fn,
        tp,
    ]

    plt.figure(figsize=(9, 6))

    plt.bar(
        labels,
        values,
    )

    plt.title(
        "Final Model Prediction Outcome Analysis"
    )

    plt.xlabel("Prediction Outcome")

    plt.ylabel("Number of Patients")

    plt.xticks(rotation=20)

    for index, value in enumerate(values):

        plt.text(
            index,
            value + 0.2,
            str(value),
            ha="center",
        )

    plt.tight_layout()

    output_path = (
        FIGURES_PATH
        / "09_prediction_error_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nError analysis figure saved to:\n"
        f"{output_path}"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)

    print(
        "BIOINSIGHT AI — "
        "SAVED MODEL EVALUATION & ERROR ANALYSIS"
    )

    print("=" * 65)

    df = load_data()

    print(f"\nDataset Shape: {df.shape}")

    X_test, y_test = create_test_set(df)

    print(f"Test Patients: {len(X_test)}")

    model = load_model()

    print(
        f"\nSaved model loaded successfully from:\n"
        f"{MODEL_PATH}"
    )

    (
        metrics,
        predictions,
        probabilities,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print_metrics(metrics)

    print_classification_report(
        y_test,
        predictions,
    )

    tn, fp, fn, tp = analyze_confusion_matrix(
        y_test,
        predictions,
    )

    save_misclassified_patients(
        X_test,
        y_test,
        predictions,
        probabilities,
    )

    save_error_analysis_figure(
        tn,
        fp,
        fn,
        tp,
    )

    print("\n" + "=" * 65)

    print(
        "SAVED MODEL EVALUATION "
        "COMPLETED SUCCESSFULLY"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()