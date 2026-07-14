"""
BioInsight-AI
Integration Tests

Tests the complete core pipeline:

Patient Data
    ↓
Validation
    ↓
Saved ML Pipeline
    ↓
Prediction
    ↓
Probability
    ↓
Preprocessing
    ↓
SHAP Explainer
    ↓
Patient-Level Explanation
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.validation import (
    create_invalid_test_patient,
    create_valid_test_patient,
    validate_patient_data,
    validate_patient_data_or_raise,
)

from src.explain import (
    create_shap_explainer,
    explain_patient,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "heart_disease_clean.csv"
)


# ============================================================
# PIPELINE FIXTURE
# ============================================================

@pytest.fixture(scope="module")
def pipeline():

    assert MODEL_PATH.exists(), (
        f"Saved model not found at: {MODEL_PATH}"
    )

    loaded_pipeline = joblib.load(MODEL_PATH)

    return loaded_pipeline


# ============================================================
# DATASET FIXTURE
# ============================================================

@pytest.fixture(scope="module")
def dataset():

    assert DATA_PATH.exists(), (
        f"Processed dataset not found at: {DATA_PATH}"
    )

    df = pd.read_csv(DATA_PATH)

    assert "target" in df.columns

    return df


# ============================================================
# TRAINING FEATURES
# ============================================================

@pytest.fixture(scope="module")
def X_train(dataset):

    return dataset.drop(columns=["target"])


# ============================================================
# PREPROCESSOR FIXTURE
# ============================================================

@pytest.fixture(scope="module")
def preprocessor(pipeline):

    assert hasattr(pipeline, "named_steps")

    assert "preprocessor" in pipeline.named_steps

    return pipeline.named_steps["preprocessor"]


# ============================================================
# MODEL FIXTURE
# ============================================================

@pytest.fixture(scope="module")
def model(pipeline):

    assert hasattr(pipeline, "named_steps")

    assert "model" in pipeline.named_steps

    return pipeline.named_steps["model"]


# ============================================================
# TRANSFORMED TRAINING DATA
# ============================================================

@pytest.fixture(scope="module")
def X_train_transformed(
    preprocessor,
    X_train,
):

    transformed = preprocessor.transform(X_train)

    if hasattr(transformed, "toarray"):

        transformed = transformed.toarray()

    return transformed


# ============================================================
# TRANSFORMED FEATURE NAMES
# ============================================================

@pytest.fixture(scope="module")
def feature_names(preprocessor):

    names = preprocessor.get_feature_names_out()

    return list(names)


# ============================================================
# SHAP EXPLAINER FIXTURE
# ============================================================

@pytest.fixture(scope="module")
def explainer(
    model,
    X_train_transformed,
):

    return create_shap_explainer(
        model,
        X_train_transformed,
    )


# ============================================================
# PATIENT FIXTURES
# ============================================================

@pytest.fixture
def valid_patient():

    return create_valid_test_patient()


@pytest.fixture
def invalid_patient():

    return create_invalid_test_patient()


# ============================================================
# VALIDATION INTEGRATION TESTS
# ============================================================

def test_valid_patient_passes_validation(
    valid_patient,
):

    is_valid, errors = validate_patient_data(
        valid_patient
    )

    assert is_valid is True

    assert errors == []


def test_invalid_patient_fails_validation(
    invalid_patient,
):

    is_valid, errors = validate_patient_data(
        invalid_patient
    )

    assert is_valid is False

    assert len(errors) > 0


def test_invalid_patient_is_blocked_before_model(
    invalid_patient,
):

    with pytest.raises(ValueError):

        validate_patient_data_or_raise(
            invalid_patient
        )


# ============================================================
# SAVED PIPELINE STRUCTURE TESTS
# ============================================================

def test_saved_pipeline_contains_preprocessor(
    pipeline,
):

    assert "preprocessor" in pipeline.named_steps


def test_saved_pipeline_contains_model(
    pipeline,
):

    assert "model" in pipeline.named_steps


def test_saved_pipeline_has_predict_method(
    pipeline,
):

    assert hasattr(pipeline, "predict")

    assert callable(pipeline.predict)


def test_saved_pipeline_has_predict_proba_method(
    pipeline,
):

    assert hasattr(pipeline, "predict_proba")

    assert callable(pipeline.predict_proba)


# ============================================================
# PATIENT → PIPELINE TEST
# ============================================================

def test_valid_patient_can_reach_pipeline(
    pipeline,
    valid_patient,
):

    validate_patient_data_or_raise(
        valid_patient
    )

    patient_df = pd.DataFrame(
        [valid_patient]
    )

    prediction = pipeline.predict(
        patient_df
    )

    assert prediction.shape == (1,)


# ============================================================
# COMPLETE PREDICTION FLOW
# ============================================================

def test_complete_prediction_flow(
    pipeline,
    valid_patient,
):

    validate_patient_data_or_raise(
        valid_patient
    )

    patient_df = pd.DataFrame(
        [valid_patient]
    )

    prediction = pipeline.predict(
        patient_df
    )

    probabilities = pipeline.predict_proba(
        patient_df
    )

    assert prediction.shape == (1,)

    assert probabilities.shape == (1, 2)

    assert prediction[0] in [0, 1]

    assert np.all(
        probabilities >= 0
    )

    assert np.all(
        probabilities <= 1
    )

    assert np.isclose(
        probabilities[0].sum(),
        1.0,
    )


# ============================================================
# PREDICTION / PROBABILITY CONSISTENCY
# ============================================================

def test_prediction_matches_probability(
    pipeline,
    valid_patient,
):

    patient_df = pd.DataFrame(
        [valid_patient]
    )

    prediction = pipeline.predict(
        patient_df
    )

    probabilities = pipeline.predict_proba(
        patient_df
    )

    expected_prediction = np.argmax(
        probabilities,
        axis=1,
    )

    assert np.array_equal(
        prediction,
        expected_prediction,
    )


# ============================================================
# PREPROCESSING INTEGRATION TESTS
# ============================================================

def test_preprocessor_transforms_dataset(
    X_train_transformed,
):

    assert X_train_transformed is not None

    assert len(X_train_transformed) > 0


def test_transformed_dataset_is_two_dimensional(
    X_train_transformed,
):

    assert X_train_transformed.ndim == 2


def test_transformed_dataset_has_samples(
    X_train_transformed,
):

    assert X_train_transformed.shape[0] > 0


def test_transformed_dataset_has_features(
    X_train_transformed,
):

    assert X_train_transformed.shape[1] > 0


def test_transformed_feature_count_matches_feature_names(
    X_train_transformed,
    feature_names,
):

    assert (
        X_train_transformed.shape[1]
        == len(feature_names)
    )


# ============================================================
# SHAP EXPLAINER TESTS
# ============================================================

def test_shap_explainer_can_be_created(
    explainer,
):

    assert explainer is not None


def test_valid_patient_can_be_explained(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert result is not None


# ============================================================
# EXPLANATION TYPE TEST
# ============================================================

def test_explanation_is_dictionary(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert isinstance(result, dict)


# ============================================================
# EXPLANATION RETURN STRUCTURE
# ============================================================

def test_explanation_contains_expected_keys(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    expected_keys = {
        "prediction",
        "prediction_label",
        "positive_class_probability",
        "predicted_class_probability",
        "patient_dataframe",
        "transformed_patient",
        "all_contributions",
        "toward_heart_disease",
        "toward_no_heart_disease",
        "shap_explanation",
    }

    assert expected_keys.issubset(
        result.keys()
    )


# ============================================================
# PREDICTION INFORMATION TESTS
# ============================================================

def test_explanation_contains_prediction(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert "prediction" in result

    assert isinstance(
        result["prediction"],
        int,
    )


def test_explanation_contains_prediction_label(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert "prediction_label" in result

    assert isinstance(
        result["prediction_label"],
        str,
    )


# ============================================================
# PROBABILITY INFORMATION TESTS
# ============================================================

def test_explanation_contains_probability_information(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert "positive_class_probability" in result

    assert "predicted_class_probability" in result

    assert isinstance(
        result["positive_class_probability"],
        float,
    )

    assert isinstance(
        result["predicted_class_probability"],
        float,
    )


def test_explanation_probabilities_are_valid(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert (
        0.0
        <= result["positive_class_probability"]
        <= 1.0
    )

    assert (
        0.0
        <= result["predicted_class_probability"]
        <= 1.0
    )


# ============================================================
# PATIENT DATAFRAME TEST
# ============================================================

def test_explanation_contains_patient_dataframe(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert "patient_dataframe" in result

    assert isinstance(
        result["patient_dataframe"],
        pd.DataFrame,
    )

    assert len(
        result["patient_dataframe"]
    ) == 1


# ============================================================
# TRANSFORMED PATIENT TEST
# ============================================================

def test_explanation_contains_transformed_patient(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert "transformed_patient" in result

    assert isinstance(
        result["transformed_patient"],
        pd.DataFrame,
    )

    assert len(
        result["transformed_patient"]
    ) == 1


# ============================================================
# CONTRIBUTION INFORMATION TEST
# ============================================================

def test_explanation_contains_contribution_information(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert "all_contributions" in result

    assert "toward_heart_disease" in result

    assert "toward_no_heart_disease" in result

    assert isinstance(
        result["all_contributions"],
        pd.DataFrame,
    )

    assert isinstance(
        result["toward_heart_disease"],
        pd.DataFrame,
    )

    assert isinstance(
        result["toward_no_heart_disease"],
        pd.DataFrame,
    )


# ============================================================
# NON-EMPTY SHAP OUTPUT TEST
# ============================================================

def test_explanation_has_non_empty_shap_output(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert "all_contributions" in result

    assert "shap_explanation" in result

    contributions = result[
        "all_contributions"
    ]

    shap_explanation = result[
        "shap_explanation"
    ]

    assert isinstance(
        contributions,
        pd.DataFrame,
    )

    assert len(contributions) > 0

    assert shap_explanation is not None


# ============================================================
# CONTRIBUTION DIRECTION TESTS
# ============================================================

def test_heart_disease_contributions_are_positive(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    contributions = result[
        "toward_heart_disease"
    ]

    assert isinstance(
        contributions,
        pd.DataFrame,
    )

    assert len(contributions) > 0

    assert "shap_value" in contributions.columns

    assert (
        contributions["shap_value"] > 0
    ).all()


def test_no_heart_disease_contributions_are_negative(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    contributions = result[
        "toward_no_heart_disease"
    ]

    assert isinstance(
        contributions,
        pd.DataFrame,
    )

    assert len(contributions) > 0

    assert "shap_value" in contributions.columns

    assert (
        contributions["shap_value"] < 0
    ).all()


# ============================================================
# EXPLANATION / PIPELINE PREDICTION CONSISTENCY
# ============================================================

def test_explanation_prediction_matches_pipeline_prediction(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    patient_df = pd.DataFrame(
        [valid_patient]
    )

    expected_prediction = int(
        pipeline.predict(
            patient_df
        )[0]
    )

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert (
        result["prediction"]
        == expected_prediction
    )


# ============================================================
# EXPLANATION / PIPELINE PROBABILITY CONSISTENCY
# ============================================================

def test_explanation_positive_probability_matches_pipeline(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    patient_df = pd.DataFrame(
        [valid_patient]
    )

    expected_probability = float(
        pipeline.predict_proba(
            patient_df
        )[0][1]
    )

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    actual_probability = result[
        "positive_class_probability"
    ]

    assert np.isclose(
        actual_probability,
        expected_probability,
    )


# ============================================================
# PREDICTED CLASS PROBABILITY CONSISTENCY
# ============================================================

def test_predicted_class_probability_matches_pipeline(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    patient_df = pd.DataFrame(
        [valid_patient]
    )

    prediction = int(
        pipeline.predict(
            patient_df
        )[0]
    )

    probabilities = pipeline.predict_proba(
        patient_df
    )[0]

    expected_probability = float(
        probabilities[prediction]
    )

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert np.isclose(
        result["predicted_class_probability"],
        expected_probability,
    )


# ============================================================
# SHAP EXPLANATION TEST
# ============================================================

def test_shap_explanation_exists(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert "shap_explanation" in result

    assert result["shap_explanation"] is not None


# ============================================================
# PIPELINE DETERMINISM
# ============================================================

def test_pipeline_prediction_is_deterministic(
    pipeline,
    valid_patient,
):

    patient_df = pd.DataFrame(
        [valid_patient]
    )

    prediction_1 = pipeline.predict(
        patient_df
    )

    prediction_2 = pipeline.predict(
        patient_df
    )

    assert np.array_equal(
        prediction_1,
        prediction_2,
    )


# ============================================================
# EXPLANATION DETERMINISM
# ============================================================

def test_explanation_prediction_is_deterministic(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result_1 = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    result_2 = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    assert (
        result_1["prediction"]
        == result_2["prediction"]
    )

    assert np.isclose(
        result_1["positive_class_probability"],
        result_2["positive_class_probability"],
    )


# ============================================================
# CONTRIBUTION DETERMINISM
# ============================================================

def test_shap_contributions_are_deterministic(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    result_1 = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    result_2 = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    contributions_1 = result_1[
        "all_contributions"
    ].reset_index(drop=True)

    contributions_2 = result_2[
        "all_contributions"
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        contributions_1,
        contributions_2,
    )


# ============================================================
# END-TO-END BIOINSIGHT-AI CORE PIPELINE TEST
# ============================================================

def test_end_to_end_bioinsight_core_pipeline(
    pipeline,
    explainer,
    valid_patient,
    feature_names,
):

    # STEP 1
    # Validate patient input

    validate_patient_data_or_raise(
        valid_patient
    )

    # STEP 2
    # Convert patient input to DataFrame

    patient_df = pd.DataFrame(
        [valid_patient]
    )

    # STEP 3
    # Generate prediction

    prediction = pipeline.predict(
        patient_df
    )

    # STEP 4
    # Generate probabilities

    probabilities = pipeline.predict_proba(
        patient_df
    )

    # STEP 5
    # Generate SHAP explanation

    explanation = explain_patient(
        pipeline,
        explainer,
        valid_patient,
        feature_names,
    )

    # STEP 6
    # Verify model output

    assert prediction.shape == (1,)

    assert probabilities.shape == (1, 2)

    assert prediction[0] in [0, 1]

    assert np.isclose(
        probabilities[0].sum(),
        1.0,
    )

    # STEP 7
    # Verify explanation output

    assert isinstance(
        explanation,
        dict,
    )

    assert (
        explanation["prediction"]
        == int(prediction[0])
    )

    assert np.isclose(
        explanation["positive_class_probability"],
        probabilities[0][1],
    )

    # STEP 8
    # Verify SHAP output

    assert isinstance(
        explanation["all_contributions"],
        pd.DataFrame,
    )

    assert len(
        explanation["all_contributions"]
    ) > 0

    assert (
        explanation["shap_explanation"]
        is not None
    )