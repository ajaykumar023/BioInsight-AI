from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.model_selection import train_test_split


# ==================================================
# PROJECT PATHS
# ==================================================

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

COEFFICIENTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "feature_coefficients.csv"
)

SHAP_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "shap_global_importance.csv"
)

PATIENT_CONTRIBUTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "patient_shap_contributions.csv"
)

FIGURES_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)


# ==================================================
# DATASET CONFIGURATION
# ==================================================

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


INPUT_FEATURES = [
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
]


# ==================================================
# LOAD SAVED PIPELINE
# ==================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Saved model not found:\n{MODEL_PATH}\n"
            "Run python src\\train.py first."
        )

    return joblib.load(MODEL_PATH)


# ==================================================
# LOAD DATA
# ==================================================

def load_data():

    if not RAW_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_DATA_PATH}"
        )

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


# ==================================================
# RECREATE TRAIN / TEST SPLIT
# ==================================================

def split_data(df):

    X = df.drop(columns=["target"])

    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


# ==================================================
# EXTRACT PIPELINE COMPONENTS
# ==================================================

def extract_pipeline_components(pipeline):

    if "preprocessor" not in pipeline.named_steps:

        raise ValueError(
            "Pipeline does not contain "
            "a 'preprocessor' step."
        )

    if "model" not in pipeline.named_steps:

        raise ValueError(
            "Pipeline does not contain "
            "a 'model' step."
        )

    return (
        pipeline.named_steps["preprocessor"],
        pipeline.named_steps["model"],
    )


# ==================================================
# GET TRANSFORMED FEATURE NAMES
# ==================================================

def get_feature_names(preprocessor):

    return preprocessor.get_feature_names_out()


# ==================================================
# GLOBAL COEFFICIENT EXPLANATION
# ==================================================

def create_coefficient_analysis(
    model,
    feature_names,
):

    if not hasattr(model, "coef_"):

        raise TypeError(
            "The selected model does not provide "
            "linear coefficients."
        )

    coefficients = model.coef_[0]

    if len(coefficients) != len(feature_names):

        raise ValueError(
            "Coefficient count does not match "
            "feature count."
        )

    coefficients_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    coefficients_df["absolute_coefficient"] = (
        coefficients_df["coefficient"].abs()
    )

    coefficients_df["effect_direction"] = (
        coefficients_df["coefficient"]
        .apply(
            lambda value: (
                "Toward Heart Disease"
                if value > 0
                else (
                    "Toward No Heart Disease"
                    if value < 0
                    else "No Effect"
                )
            )
        )
    )

    return (
        coefficients_df
        .sort_values(
            by="absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ==================================================
# SAVE COEFFICIENT OUTPUTS
# ==================================================

def save_coefficient_outputs(
    coefficients_df,
    top_n=15,
):

    COEFFICIENTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    coefficients_df.to_csv(
        COEFFICIENTS_PATH,
        index=False,
    )

    top_features = (
        coefficients_df
        .head(top_n)
        .sort_values(
            by="coefficient",
            ascending=True,
        )
    )

    plt.figure(figsize=(11, 8))

    plt.barh(
        top_features["feature"],
        top_features["coefficient"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.title(
        "Top Global Logistic Regression Coefficients"
    )

    plt.xlabel("Coefficient Value")

    plt.ylabel("Transformed Feature")

    plt.tight_layout()

    output_path = (
        FIGURES_PATH
        / "10_global_model_coefficients.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==================================================
# TRANSFORM DATA
# ==================================================

def transform_data(
    preprocessor,
    X_train,
    X_test,
    feature_names,
):

    X_train_transformed = (
        preprocessor.transform(X_train)
    )

    X_test_transformed = (
        preprocessor.transform(X_test)
    )

    if hasattr(X_train_transformed, "toarray"):

        X_train_transformed = (
            X_train_transformed.toarray()
        )

    if hasattr(X_test_transformed, "toarray"):

        X_test_transformed = (
            X_test_transformed.toarray()
        )

    X_train_transformed_df = pd.DataFrame(
        X_train_transformed,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_transformed_df = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index,
    )

    return (
        X_train_transformed_df,
        X_test_transformed_df,
    )


# ==================================================
# CREATE SHAP EXPLAINER
# ==================================================

def create_shap_explainer(
    model,
    X_train_transformed,
):

    # SHAP explainer for SVM
    background = shap.sample(
        X_train_transformed,
        100,
        random_state=42,
    )

    return shap.KernelExplainer(
        model.predict_proba,
        background,
    )

# ==================================================
# GLOBAL SHAP ANALYSIS
# ==================================================

def create_global_shap_analysis(
    explainer,
    X_test_transformed,
):

    shap_values = explainer(
        X_test_transformed
    )

    mean_absolute_shap = (
        abs(shap_values.values)
        .mean(axis=0)
    )

    shap_importance_df = pd.DataFrame(
        {
            "feature": (
                X_test_transformed.columns
            ),
            "mean_absolute_shap": (
                mean_absolute_shap
            ),
        }
    )

    shap_importance_df = (
        shap_importance_df
        .sort_values(
            by="mean_absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return (
        shap_values,
        shap_importance_df,
    )


# ==================================================
# SAVE GLOBAL SHAP OUTPUTS
# ==================================================

def save_global_shap_outputs(
    shap_values,
    shap_importance_df,
):

    SHAP_IMPORTANCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    shap_importance_df.to_csv(
        SHAP_IMPORTANCE_PATH,
        index=False,
    )

    shap.plots.bar(
        shap_values,
        max_display=15,
        show=False,
    )

    plt.title(
        "Global SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH
        / "11_global_shap_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    shap.plots.beeswarm(
        shap_values,
        max_display=15,
        show=False,
    )

    plt.title(
        "Global SHAP Summary Plot"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_PATH
        / "12_global_shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==================================================
# VALIDATE PATIENT INPUT
# ==================================================

def validate_patient_input(patient_data):

    if not isinstance(patient_data, dict):

        raise TypeError(
            "patient_data must be a dictionary."
        )

    missing_features = [
        feature
        for feature in INPUT_FEATURES
        if feature not in patient_data
    ]

    extra_features = [
        feature
        for feature in patient_data
        if feature not in INPUT_FEATURES
    ]

    if missing_features:

        raise ValueError(
            "Missing patient features: "
            + ", ".join(missing_features)
        )

    if extra_features:

        raise ValueError(
            "Unexpected patient features: "
            + ", ".join(extra_features)
        )

    patient_df = pd.DataFrame(
        [patient_data],
        columns=INPUT_FEATURES,
    )

    return patient_df


# ==================================================
# EXPLAIN ANY PATIENT
# ==================================================

def explain_patient(
    pipeline,
    explainer,
    patient_data,
    feature_names,
    top_n=10,
):

    patient_df = validate_patient_input(
        patient_data
    )

    prediction = int(
        pipeline.predict(patient_df)[0]
    )

    class_probabilities = (
        pipeline.predict_proba(patient_df)[0]
    )

    positive_class_probability = float(
        class_probabilities[1]
    )

    predicted_class_probability = float(
        class_probabilities[prediction]
    )

    preprocessor = (
        pipeline.named_steps["preprocessor"]
    )

    transformed_patient = (
        preprocessor.transform(patient_df)
    )

    if hasattr(transformed_patient, "toarray"):

        transformed_patient = (
            transformed_patient.toarray()
        )

    transformed_patient_df = pd.DataFrame(
        transformed_patient,
        columns=feature_names,
    )

    # Get SHAP values using KernelExplainer
    shap_values = explainer.shap_values(
        transformed_patient_df,
        nsamples=100
    )

    # Binary classification
    import numpy as np

    if isinstance(shap_values, list):
        patient_shap_values = shap_values[1][0]

    else:
        patient_shap_values = np.array(shap_values)

        # (25,2) -> take class 1
        if patient_shap_values.ndim == 2:
            patient_shap_values = patient_shap_values[:, 1]

        # (1,25,2)
        elif patient_shap_values.ndim == 3:
            patient_shap_values = patient_shap_values[0, :, 1]
    
    contributions_df = pd.DataFrame({
        "feature": feature_names,
        "transformed_value": transformed_patient_df.iloc[0].values,
        "shap_value": patient_shap_values,
    }
    )

    contributions_df["absolute_shap"] = (
        contributions_df["shap_value"].abs()
    )

    contributions_df["effect_direction"] = (
        contributions_df["shap_value"]
        .apply(
            lambda value: (
                "Toward Heart Disease"
                if value > 0
                else (
                    "Toward No Heart Disease"
                    if value < 0
                    else "No Contribution"
                )
            )
        )
    )

    contributions_df = (
        contributions_df
        .sort_values(
            by="absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    toward_positive_class = (
        contributions_df[
            contributions_df["shap_value"] > 0
        ]
        .head(top_n)
        .reset_index(drop=True)
    )

    toward_negative_class = (
        contributions_df[
            contributions_df["shap_value"] < 0
        ]
        .head(top_n)
        .reset_index(drop=True)
    )

    result = {
        "prediction": prediction,
        "prediction_label": (
            "Heart Disease"
            if prediction == 1
            else "No Heart Disease"
        ),
        "positive_class_probability": (
            positive_class_probability
        ),
        "predicted_class_probability": (
            predicted_class_probability
        ),
        "patient_dataframe": patient_df,
        "transformed_patient": (
            transformed_patient_df
        ),
        "all_contributions": (
            contributions_df
        ),
        "toward_heart_disease": (
            toward_positive_class
        ),
        "toward_no_heart_disease": (
            toward_negative_class
        ),
        "shap_explanation": patient_shap_values,
    }

    return result


# ==================================================
# SAVE DEMO PATIENT EXPLANATION
# ==================================================

def save_demo_patient_outputs(result):

    PATIENT_CONTRIBUTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    result["all_contributions"].to_csv(
        PATIENT_CONTRIBUTIONS_PATH,
        index=False,
    )

    shap.plots.waterfall(
        result["shap_explanation"],
        max_display=15,
        show=False,
    )

    plt.title(
        "Reusable Patient-Level SHAP Explanation"
    )

    plt.tight_layout()

    output_path = (
        FIGURES_PATH
        / "13_patient_shap_waterfall.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ==================================================
# PRINT PATIENT EXPLANATION
# ==================================================

def print_patient_explanation(
    result,
    top_n=5,
):

    print("\n" + "=" * 70)

    print("REUSABLE PATIENT EXPLANATION ENGINE")

    print("=" * 70)

    print(
        f"\nPrediction                 : "
        f"{result['prediction_label']}"
    )

    print(
        f"Positive Class Probability : "
        f"{result['positive_class_probability']:.4f}"
    )

    print(
        f"Predicted Class Confidence : "
        f"{result['predicted_class_probability']:.4f}"
    )

    print("\n" + "-" * 70)

    print("TOP FEATURES TOWARD HEART DISEASE")

    print("-" * 70)

    positive_features = (
        result["toward_heart_disease"]
        .head(top_n)
    )

    if positive_features.empty:

        print("\nNo positive SHAP contributions.")

    else:

        print(
            "\n"
            + positive_features[
                [
                    "feature",
                    "transformed_value",
                    "shap_value",
                ]
            ].to_string(index=False)
        )

    print("\n" + "-" * 70)

    print("TOP FEATURES TOWARD NO HEART DISEASE")

    print("-" * 70)

    negative_features = (
        result["toward_no_heart_disease"]
        .head(top_n)
    )

    if negative_features.empty:

        print("\nNo negative SHAP contributions.")

    else:

        print(
            "\n"
            + negative_features[
                [
                    "feature",
                    "transformed_value",
                    "shap_value",
                ]
            ].to_string(index=False)
        )


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 70)

    print(
        "BIOINSIGHT AI — "
        "REUSABLE EXPLAINABILITY ENGINE"
    )

    print("=" * 70)

    # --------------------------------------------------
    # LOAD PIPELINE AND DATA
    # --------------------------------------------------

    pipeline = load_model()

    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(df)

    print(
        f"\nTraining Patients: "
        f"{len(X_train)}"
    )

    print(
        f"Testing Patients : "
        f"{len(X_test)}"
    )

    # --------------------------------------------------
    # PIPELINE COMPONENTS
    # --------------------------------------------------

    (
        preprocessor,
        model,
    ) = extract_pipeline_components(
        pipeline
    )

    feature_names = get_feature_names(
        preprocessor
    )

    print(
        f"\nTransformed Features: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------
    # GLOBAL COEFFICIENT ANALYSIS
    # --------------------------------------------------

    coefficients_df = (
        create_coefficient_analysis(
            model,
            feature_names,
        )
    )

    save_coefficient_outputs(
        coefficients_df
    )

    print(
        "\nGlobal coefficient analysis completed."
    )

    # --------------------------------------------------
    # TRANSFORM DATA
    # --------------------------------------------------

    (
        X_train_transformed,
        X_test_transformed,
    ) = transform_data(
        preprocessor,
        X_train,
        X_test,
        feature_names,
    )

    # --------------------------------------------------
    # CREATE SHAP EXPLAINER
    # --------------------------------------------------

    explainer = create_shap_explainer(
        model,
        X_train_transformed,
    )

    # --------------------------------------------------
    # GLOBAL SHAP
    # --------------------------------------------------

    (
        shap_values,
        shap_importance_df,
    ) = create_global_shap_analysis(
        explainer,
        X_test_transformed,
    )

    save_global_shap_outputs(
        shap_values,
        shap_importance_df,
    )

    print(
        "Global SHAP analysis completed."
    )

    # --------------------------------------------------
    # DEMO PATIENT
    #
    # We use the first held-out test patient only to
    # verify that the reusable function works.
    # The web app will later pass user-entered values.
    # --------------------------------------------------

    demo_patient = (
        X_test.iloc[0].to_dict()
    )

    result = explain_patient(
        pipeline=pipeline,
        explainer=explainer,
        patient_data=demo_patient,
        feature_names=feature_names,
        top_n=10,
    )

    print_patient_explanation(
        result,
        top_n=5,
    )

    save_demo_patient_outputs(
        result
    )

    # --------------------------------------------------
    # VERIFY RETURNED OBJECTS
    # --------------------------------------------------

    print("\n" + "=" * 70)

    print("ENGINE OUTPUT VERIFICATION")

    print("=" * 70)

    print(
        f"\nPrediction Type: "
        f"{type(result['prediction']).__name__}"
    )

    print(
        f"Contribution Rows: "
        f"{len(result['all_contributions'])}"
    )

    print(
        f"Positive Contribution Rows: "
        f"{len(result['toward_heart_disease'])}"
    )

    print(
        f"Negative Contribution Rows: "
        f"{len(result['toward_no_heart_disease'])}"
    )

    print(
        f"SHAP Explanation Type: "
        f"{type(result['shap_explanation']).__name__}"
    )

    # --------------------------------------------------
    # COMPLETE
    # --------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "DAY 4 PART 4 "
        "COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()