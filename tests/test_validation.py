"""
BioInsight-AI
Automated Tests for Clinical Input Validation
"""

import pytest

from src.validation import (
    REQUIRED_FEATURES,
    create_invalid_test_patient,
    create_valid_test_patient,
    validate_patient_data,
    validate_patient_data_or_raise,
)


# ============================================================
# VALID PATIENT TESTS
# ============================================================

def test_valid_patient_passes_validation():

    patient = create_valid_test_patient()

    is_valid, errors = validate_patient_data(patient)

    assert is_valid is True
    assert errors == []


def test_valid_patient_contains_all_required_features():

    patient = create_valid_test_patient()

    assert set(patient.keys()) == set(REQUIRED_FEATURES)


def test_valid_patient_does_not_raise_exception():

    patient = create_valid_test_patient()

    validate_patient_data_or_raise(patient)


# ============================================================
# INVALID PATIENT TESTS
# ============================================================

def test_invalid_patient_fails_validation():

    patient = create_invalid_test_patient()

    is_valid, errors = validate_patient_data(patient)

    assert is_valid is False
    assert len(errors) > 0


def test_invalid_patient_raises_value_error():

    patient = create_invalid_test_patient()

    with pytest.raises(ValueError):
        validate_patient_data_or_raise(patient)


# ============================================================
# REQUIRED FEATURE TESTS
# ============================================================

@pytest.mark.parametrize(
    "feature",
    REQUIRED_FEATURES,
)
def test_missing_required_feature_is_rejected(feature):

    patient = create_valid_test_patient()

    del patient[feature]

    is_valid, errors = validate_patient_data(patient)

    assert is_valid is False
    assert len(errors) >= 1

    assert any(
        "Missing required feature" in error
        for error in errors
    )


# ============================================================
# NUMERICAL RANGE TESTS
# ============================================================

@pytest.mark.parametrize(
    "feature, invalid_value",
    [
        ("age", 0),
        ("age", 121),
        ("resting_blood_pressure", 49),
        ("resting_blood_pressure", 251),
        ("cholesterol", 49),
        ("cholesterol", 701),
        ("max_heart_rate", 29),
        ("max_heart_rate", 251),
        ("st_depression", -0.1),
        ("st_depression", 10.1),
    ],
)
def test_invalid_numerical_ranges_are_rejected(
    feature,
    invalid_value,
):

    patient = create_valid_test_patient()

    patient[feature] = invalid_value

    is_valid, errors = validate_patient_data(patient)

    assert is_valid is False
    assert len(errors) >= 1


# ============================================================
# NUMERICAL TYPE TESTS
# ============================================================

@pytest.mark.parametrize(
    "feature",
    [
        "age",
        "resting_blood_pressure",
        "cholesterol",
        "max_heart_rate",
        "st_depression",
    ],
)
def test_non_numeric_values_are_rejected(feature):

    patient = create_valid_test_patient()

    patient[feature] = "invalid"

    is_valid, errors = validate_patient_data(patient)

    assert is_valid is False

    assert any(
        "must be a numeric value" in error
        for error in errors
    )


def test_boolean_is_not_accepted_as_numeric_value():

    patient = create_valid_test_patient()

    patient["age"] = True

    is_valid, errors = validate_patient_data(patient)

    assert is_valid is False
    assert len(errors) >= 1


# ============================================================
# CATEGORICAL VALUE TESTS
# ============================================================

@pytest.mark.parametrize(
    "feature, invalid_value",
    [
        ("sex", 5),
        ("chest_pain_type", 10),
        ("fasting_blood_sugar", 3),
        ("resting_ecg", 8),
        ("exercise_induced_angina", 4),
        ("st_slope", 9),
        ("major_vessels", 10),
        ("thalassemia", 100),
    ],
)
def test_invalid_categorical_values_are_rejected(
    feature,
    invalid_value,
):

    patient = create_valid_test_patient()

    patient[feature] = invalid_value

    is_valid, errors = validate_patient_data(patient)

    assert is_valid is False
    assert len(errors) >= 1


# ============================================================
# UNKNOWN FEATURE TESTS
# ============================================================

def test_unknown_feature_is_rejected_by_default():

    patient = create_valid_test_patient()

    patient["unknown_feature"] = 123

    is_valid, errors = validate_patient_data(patient)

    assert is_valid is False

    assert any(
        "Unknown patient feature received" in error
        for error in errors
    )


def test_unknown_feature_can_be_allowed():

    patient = create_valid_test_patient()

    patient["unknown_feature"] = 123

    is_valid, errors = validate_patient_data(
        patient,
        reject_unknown_features=False,
    )

    assert is_valid is True
    assert errors == []


# ============================================================
# INPUT TYPE TESTS
# ============================================================

@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        [],
        (),
        "patient",
        123,
    ],
)
def test_non_dictionary_patient_data_is_rejected(invalid_input):

    is_valid, errors = validate_patient_data(invalid_input)

    assert is_valid is False
    assert len(errors) == 1

    assert (
        errors[0]
        == "Patient data must be provided as a dictionary."
    )


# ============================================================
# MULTIPLE ERROR TEST
# ============================================================

def test_multiple_validation_errors_are_returned_together():

    patient = create_valid_test_patient()

    patient["age"] = 200
    patient["sex"] = 9
    patient["cholesterol"] = 1000

    is_valid, errors = validate_patient_data(patient)

    assert is_valid is False
    assert len(errors) == 3