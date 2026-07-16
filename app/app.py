import plotly.graph_objects as go
import time
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ==================================================
# PROJECT PATH CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==================================================
# IMPORT BIOINSIGHT-AI BACKEND
# ==================================================

from src.explain import (
    create_shap_explainer,
    explain_patient,
    extract_pipeline_components,
    get_feature_names,
    load_data,
    load_model,
    split_data,
    transform_data,
)


# ==================================================
# STREAMLIT PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="BioInsight-AI",
    page_icon="🧬",
    layout="wide",
)
# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
<style>

/* Main background */
.stApp{
    background-color:#0E1117;
}

/* Metric cards */
div[data-testid="metric-container"]{
    background:#1B1F2A;
    border:1px solid #31333F;
    padding:18px;
    border-radius:14px;
    box-shadow:0 4px 15px rgba(0,0,0,0.25);
}

/* Buttons */
.stButton>button{
    width:100%;
    border-radius:10px;
    font-weight:bold;
}

/* Download button */
.stDownloadButton>button{
    width:100%;
    border-radius:10px;
    font-weight:bold;
}

/* Expanders */
.streamlit-expanderHeader{
    font-size:18px;
    font-weight:600;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#171A23;
}

/* Tables */
[data-testid="stDataFrame"]{
    border-radius:10px;
}

/* Headings */
h1,h2,h3{
    color:white;
}

/* Success boxes */
div[data-baseweb="notification"]{
    border-radius:12px;
}

</style>
""",
    unsafe_allow_html=True,
)
# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("# 🧬 BioInsight-AI")
    st.caption("Version 1.0")

    st.divider()

    st.subheader("🧠 AI Engine")

    st.success("✅ Logistic Regression")
    st.success("✅ SHAP Explainability")

    st.metric(
        label="Model Accuracy",
        value="86.9%"
    )

    st.metric(
        label="Dataset Size",
        value="303 Patients"
    )
    st.sidebar.markdown("---")

    st.sidebar.success("✅ AI Engine Ready")

    st.sidebar.markdown(
    """
    ### 📌 About

    🧬 **BioInsight-AI** combines Machine Learning and SHAP Explainability to assist in heart disease risk prediction.

    **Features**
    - ❤️ Heart Disease Prediction
    - 🧠 SHAP Explainability
    - 📄 PDF Report Generation
    - 📊 Risk Gauge
    - 💡 Clinical Recommendations

    ---
    ⚠️ Educational & Research Use Only.
    """
    )

# ==================================================
# HUMAN-READABLE FEATURE NAMES
# ==================================================

def format_feature_name(feature_name):

    feature_name = str(feature_name)

    replacements = {
        "numerical__age": "Age",

        "numerical__resting_blood_pressure":
            "Resting Blood Pressure",

        "numerical__cholesterol":
            "Cholesterol",

        "numerical__max_heart_rate":
            "Maximum Heart Rate",

        "numerical__st_depression":
            "ST Depression",

        "numerical__major_vessels":
            "Number of Major Vessels",

        "categorical__sex_0.0":
            "Sex: Female",

        "categorical__sex_1.0":
            "Sex: Male",

        "categorical__chest_pain_type_1.0":
            "Chest Pain: Typical Angina",

        "categorical__chest_pain_type_2.0":
            "Chest Pain: Atypical Angina",

        "categorical__chest_pain_type_3.0":
            "Chest Pain: Non-Anginal Pain",

        "categorical__chest_pain_type_4.0":
            "Chest Pain: Asymptomatic",

        "categorical__fasting_blood_sugar_0.0":
            "Fasting Blood Sugar ≤ 120 mg/dL",

        "categorical__fasting_blood_sugar_1.0":
            "Fasting Blood Sugar > 120 mg/dL",

        "categorical__resting_ecg_0.0":
            "Resting ECG: Normal",

        "categorical__resting_ecg_1.0":
            "Resting ECG: ST-T Abnormality",

        "categorical__resting_ecg_2.0":
            "Resting ECG: Ventricular Hypertrophy",

        "categorical__exercise_induced_angina_0.0":
            "Exercise-Induced Angina: No",

        "categorical__exercise_induced_angina_1.0":
            "Exercise-Induced Angina: Yes",

        "categorical__st_slope_1.0":
            "ST Slope: Upsloping",

        "categorical__st_slope_2.0":
            "ST Slope: Flat",

        "categorical__st_slope_3.0":
            "ST Slope: Downsloping",

        "categorical__thalassemia_3.0":
            "Thalassemia: Normal",

        "categorical__thalassemia_6.0":
            "Thalassemia: Fixed Defect",

        "categorical__thalassemia_7.0":
            "Thalassemia: Reversible Defect",
    }

    if feature_name in replacements:
        return replacements[feature_name]

    readable_name = (
        feature_name
        .replace("numerical__", "")
        .replace("categorical__", "")
        .replace("_", " ")
        .title()
    )

    return readable_name


# ==================================================
# PREPARE SHAP CONTRIBUTION DATA
# ==================================================

def prepare_contribution_dataframe(contributions):

    if contributions is None:
        return pd.DataFrame()

    if isinstance(contributions, pd.DataFrame):
        df = contributions.copy()

    else:
        df = pd.DataFrame(contributions)

    if df.empty:
        return df

    required_columns = {
        "feature",
        "shap_value",
    }

    if not required_columns.issubset(df.columns):
        return pd.DataFrame()

    df["feature"] = df["feature"].astype(str)

    df["display_feature"] = (
        df["feature"]
        .apply(format_feature_name)
    )

    df["shap_value"] = pd.to_numeric(
        df["shap_value"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["shap_value"]
    )

    df["absolute_shap"] = (
        df["shap_value"].abs()
    )

    return df


# ==================================================
# DISPLAY SHAP FACTOR TABLE
# ==================================================

def display_factor_table(
    dataframe,
    title,
    direction,
):

    st.subheader(title)

    if dataframe.empty:

        st.info(
            "No significant factors were available "
            "for this direction."
        )

        return

    display_df = dataframe[
        [
            "display_feature",
            "shap_value",
        ]
    ].copy()

    display_df.columns = [
        "Clinical Factor",
        "SHAP Contribution",
    ]

    display_df[
        "SHAP Contribution"
    ] = display_df[
        "SHAP Contribution"
    ].round(4)

    if direction == "positive":

        st.warning(
            "Positive SHAP values push the model's "
            "prediction toward the Heart Disease class."
        )

    else:

        st.success(
            "Negative SHAP values push the model's "
            "prediction toward the No Heart Disease class."
        )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )


# ==================================================
# DISPLAY CONTRIBUTION CHART
# ==================================================

def display_contribution_chart(
    dataframe,
    title,
):

    if dataframe.empty:
        return

    chart_df = dataframe[
        [
            "display_feature",
            "shap_value",
        ]
    ].copy()

    chart_df = chart_df.set_index(
        "display_feature"
    )

    chart_df.columns = [
        "SHAP Contribution"
    ]

    st.subheader(title)

    st.bar_chart(
        chart_df,
        horizontal=True,
        width="stretch",
    )


# ==================================================
# MODEL OUTPUT BAND
# ==================================================

def get_model_output_band(probability):

    if probability < 0.30:
        return "Lower Model-Estimated Probability"

    elif probability < 0.70:
        return "Intermediate Model-Estimated Probability"

    else:
        return "Higher Model-Estimated Probability"


# ==================================================
# BUILD DOWNLOADABLE PATIENT REPORT
# ==================================================

def build_patient_pdf(
    patient_data,
    prediction_label,
    heart_disease_probability,
    predicted_confidence,
    output_band,
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    title = styles["Heading1"]
    title.alignment = TA_CENTER

    story = []

    story.append(
        Paragraph(
            "BioInsight-AI Patient Report",
            title,
        )
    )

    story.append(
        Spacer(1,0.3*inch)
    )

    story.append(
        Paragraph(
            "<b>Prediction:</b> "
            + prediction_label,
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            "<b>Heart Disease Probability:</b> "
            + f"{heart_disease_probability:.1%}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            "<b>Prediction Confidence:</b> "
            + f"{predicted_confidence:.1%}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            "<b>Model Output Band:</b> "
            + output_band,
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1,0.25*inch)
    )

    data = [
        ["Feature","Value"]
    ]

    for k,v in patient_data.items():

        data.append(
            [
                k.replace("_"," ").title(),
                str(v),
            ]
        )

    table = Table(data)

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND",(0,0),(-1,0),colors.darkblue),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("GRID",(0,0),(-1,-1),1,colors.grey),
                ("BACKGROUND",(0,1),(-1,-1),colors.beige),
                ("BOTTOMPADDING",(0,0),(-1,0),10),
            ]
        )
    )

    story.append(table)

    story.append(
        Spacer(1,0.3*inch)
    )

    story.append(
        Paragraph(
            "<b>Disclaimer</b>",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            "BioInsight-AI is an educational AI project. "
            "This report must not replace medical diagnosis.",
            styles["BodyText"],
        )
    )

    doc.build(story)

    pdf = buffer.getvalue()

    buffer.close()

    return pdf

# ==================================================
# LOAD AI ENGINE
# ==================================================

@st.cache_resource
def load_ai_engine():

    pipeline = load_model()

    df = load_data()

    X_train, X_test, _, _ = split_data(df)

    preprocessor, model = (
        extract_pipeline_components(
            pipeline
        )
    )

    feature_names = get_feature_names(
        preprocessor
    )

    X_train_transformed, _ = transform_data(
        preprocessor,
        X_train,
        X_test,
        feature_names,
    )

    explainer = create_shap_explainer(
        model,
        X_train_transformed,
    )

    return (
        pipeline,
        explainer,
        feature_names,
    )


# ==================================================
# INITIALIZE AI ENGINE
# ==================================================

try:

    (
        pipeline,
        explainer,
        feature_names,
    ) = load_ai_engine()

    engine_ready = True

except Exception as error:

    pipeline = None
    explainer = None
    feature_names = None

    engine_ready = False

    st.error(
        "The BioInsight-AI prediction engine "
        "could not be loaded."
    )

    st.exception(error)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
# 🧬 BioInsight-AI

### Explainable Artificial Intelligence for Heart Disease Prediction

A clinical decision-support application powered by **Machine Learning**
and **SHAP Explainability**.

---
"""
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
    "Model",
    "Logistic"
)

with col2:
    st.metric(
        "Accuracy",
        "86.9%"
    )

with col3:
    st.metric(
        "Dataset",
        "303 Patients"
    )

with col4:
    st.metric(
        "Explainability",
        "SHAP AI"
    )

st.info(
    """
This application predicts the likelihood of heart disease and explains
which clinical features influenced the prediction.

⚠️ Educational use only. It is **not** a substitute for professional
medical diagnosis.
"""
)

st.divider()


# ==================================================
# ENGINE STATUS
# ==================================================

if engine_ready:

    st.success(
        "AI prediction and explainability engine "
        "loaded successfully."
    )

else:

    st.warning(
        "The prediction engine is currently unavailable."
    )


# ==================================================
# PATIENT INPUT FORM
# ==================================================

st.divider()

with st.form("patient_form"):
    st.header("🩺 Patient Clinical Information")

    st.caption(
    "Enter the patient's clinical measurements below. "
    "These values will be used by the trained AI model to estimate heart disease probability."
    )

    st.divider()


        # ==============================================
        # BASIC INFORMATION
        # ==============================================

    st.subheader("👤 Patient Demographics")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=50,
            step=1,
            )

    with col2:

        sex_label = st.selectbox(
            "Sex",
            options=[
                "Female",
                "Male",
                ],
            )

    with col3:

        chest_pain_label = st.selectbox(
            "Chest Pain Type",
            options=[
                "Typical Angina",
                "Atypical Angina",
                "Non-Anginal Pain",
                "Asymptomatic",
            ],
        )


    # ==============================================
    # HEART MEASUREMENTS
    # ==============================================

    st.subheader("❤️ Heart Measurements")
    col1, col2, col3 = st.columns(3)

    with col1:

        resting_blood_pressure = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=70,
            max_value=250,
            value=120,
            step=1,
        )

    with col2:

        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=80,
            max_value=700,
            value=200,
            step=1,
        )

    with col3:

        max_heart_rate = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=250,
            value=150,
            step=1,
        )


    # ==============================================
    # BLOOD SUGAR AND ECG
    # ==============================================

    st.divider()

    st.subheader("🩸 Blood Sugar & ECG")
    col1, col2 = st.columns(2)

    with col1:

        fasting_blood_sugar_label = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            options=[
                "No",
                "Yes",
            ],
        )

    with col2:

        resting_ecg_label = st.selectbox(
            "Resting ECG Result",
            options=[
                "Normal",
                "ST-T Wave Abnormality",
                "Left Ventricular Hypertrophy",
            ],
        )


    # ==============================================
    # EXERCISE INFORMATION
    # ==============================================

    st.divider()

    st.subheader("🏃 Exercise Information")
    col1, col2, col3 = st.columns(3)

    with col1:

        exercise_angina_label = st.selectbox(
            "Exercise-Induced Angina",
            options=[
                "No",
                "Yes",
            ],
        )

    with col2:

        st_depression = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

    with col3:

        st_slope_label = st.selectbox(
            "ST Segment Slope",
            options=[
                "Upsloping",
                "Flat",
                "Downsloping",
            ],
        )


    # ==============================================
    # ADVANCED CLINICAL FEATURES
    # ==============================================

    st.divider()

    st.subheader("🧬 Advanced Clinical Features")
    col1, col2 = st.columns(2)

    with col1:

        major_vessels = st.selectbox(
            "Number of Major Vessels",
            options=[
                0,
                1,
                2,
                3,
            ],
        )

    with col2:

        thalassemia_label = st.selectbox(
            "Thalassemia",
            options=[
                "Normal",
                "Fixed Defect",
                "Reversible Defect",
            ],
        )


    # ==============================================
    # SUBMIT BUTTON
    # ==============================================

    st.divider()
    st.info(
    "Please verify that all patient measurements are entered correctly before running the AI analysis."
    )
    submitted = st.form_submit_button(
        label="🔍 Analyze Patient",
        width="stretch",
        type="primary",
    )


# ==================================================
# DATASET VALUE MAPPINGS
# ==================================================

sex_map = {
    "Female": 0,
    "Male": 1,
}


chest_pain_map = {
    "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-Anginal Pain": 3,
    "Asymptomatic": 4,
}


fasting_blood_sugar_map = {
    "No": 0,
    "Yes": 1,
}


resting_ecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2,
}


exercise_angina_map = {
    "No": 0,
    "Yes": 1,
}


st_slope_map = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3,
}


thalassemia_map = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7,
}


# ==================================================
# ANALYZE PATIENT
# ==================================================

if submitted:

    patient_data = {

        "age":
            float(age),

        "sex":
            sex_map[
                sex_label
            ],

        "chest_pain_type":
            chest_pain_map[
                chest_pain_label
            ],

        "resting_blood_pressure":
            float(
                resting_blood_pressure
            ),

        "cholesterol":
            float(
                cholesterol
            ),

        "fasting_blood_sugar":
            fasting_blood_sugar_map[
                fasting_blood_sugar_label
            ],

        "resting_ecg":
            resting_ecg_map[
                resting_ecg_label
            ],

        "max_heart_rate":
            float(
                max_heart_rate
            ),

        "exercise_induced_angina":
            exercise_angina_map[
                exercise_angina_label
            ],

        "st_depression":
            float(
                st_depression
            ),

        "st_slope":
            st_slope_map[
                st_slope_label
            ],

        "major_vessels":
            int(
                major_vessels
            ),

        "thalassemia":
            thalassemia_map[
                thalassemia_label
            ],
    }


    # ==============================================
    # PATIENT SUMMARY
    # ==============================================

    st.divider()

    st.header(
        "Patient Summary"
    )

    summary_col1, summary_col2, summary_col3 = (
        st.columns(3)
    )


    with summary_col1:

        st.metric(
            "Age",
            f"{int(age)} years",
        )

        st.write(
            "**Sex:**",
            sex_label,
        )


    with summary_col2:

        st.metric(
            "Resting Blood Pressure",
            f"{int(resting_blood_pressure)} mm Hg",
        )

        st.write(
            "**Chest Pain Type:**",
            chest_pain_label,
        )


    with summary_col3:

        st.metric(
            "Maximum Heart Rate",
            int(max_heart_rate),
        )

        st.write(
            "**Thalassemia:**",
            thalassemia_label,
        )


    # ==============================================
    # RESULTS
    # ==============================================

    st.divider()

    st.header(
        "Patient Analysis Results"
    )


    if not engine_ready:

        st.error(
            "Prediction cannot run because "
            "the AI engine is unavailable."
        )


    else:

        try:
            import time

            status = st.empty()

            status.info("🧠 Initializing BioInsight-AI...")
            time.sleep(0.4)

            status.info("📂 Loading trained Logistic Regression model...")
            time.sleep(0.4)

            status.info("🧪 Validating patient information...")
            time.sleep(0.4)

            status.info("⚙️ Preprocessing clinical features...")
            time.sleep(0.4)

            status.info("❤️ Estimating heart disease probability...")
            time.sleep(0.4)

            status.info("🧬 Computing SHAP explanations...")
            time.sleep(0.5)

            status.info("📄 Preparing clinical report...")
            time.sleep(0.5)

            result = explain_patient(
                pipeline=pipeline,
                explainer=explainer,
                patient_data=patient_data,
                feature_names=feature_names,
                top_n=10,
            )

            status.success("✅ Analysis Complete!")
            time.sleep(0.5)

            status.empty()


            st.success(
                "Patient analysis completed successfully."
            )


            # ======================================
            # EXTRACT PREDICTION RESULTS
            # ======================================

            prediction = result[
                "prediction"
            ]

            prediction_label = result[
                "prediction_label"
            ]

            heart_disease_probability = result[
                "positive_class_probability"
            ]

            predicted_class_probability = result[
                "predicted_class_probability"
            ]

            # ======================================
            # AI PREDICTION RESULT
            # ======================================
            st.subheader("AI Prediction Dashboard")

            col1, col2, col3 = st.columns(3)

            # ==========================================
            # PREDICTION
            # ==========================================

            with col1:

                if prediction == 1:
                    st.error("❤️ Heart Disease Detected")
                else:
                    st.success("💚 No Heart Disease")

                st.metric(
                    label="Prediction",
                    value=prediction_label,
                )

            # ==========================================
            # PROBABILITY
            # ==========================================

            with col2:

                st.metric(
                    label="Risk Probability",
                    value=f"{heart_disease_probability:.1%}",
                )

                if heart_disease_probability < 0.30:
                    st.success("🟢 Low Risk")

                elif heart_disease_probability < 0.70:
                    st.warning("🟡 Moderate Risk")

                else:
                    st.error("🔴 High Risk")

                # ==================================================
                # HEART DISEASE RISK GAUGE
                # ==================================================

                st.markdown("<br>", unsafe_allow_html=True)

                left, center, right = st.columns([1,5,1])

                with center:

                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=heart_disease_probability * 100,
                            number={"suffix": "%"},
                            title={"text": "Estimated Risk"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#E53935"},
                                "steps": [
                                    {"range": [0, 30], "color": "#4CAF50"},
                                    {"range": [30, 70], "color": "#FFC107"},
                                    {"range": [70, 100], "color": "#F44336"},
                                ],
                                "threshold": {
                                    "line": {"color": "white", "width": 4},
                                    "thickness": 0.8,
                                    "value": heart_disease_probability * 100,
                                },
                            },
                        )
                    )

                    gauge.update_layout(
                        paper_bgcolor="#0E1117",
                        font=dict(color="white", size=18),
                        height=340,
                        margin=dict(l=20, r=20, t=40, b=10),
                    )

                    st.markdown(
                        """
                        <div style="text-align:center; margin-bottom:10px;">
                            <h2>🎯 Heart Disease Risk Gauge</h2>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.plotly_chart(
                        gauge,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
            st.divider()

            st.subheader("🩺 Clinical Recommendation")
            if heart_disease_probability < 0.30:

                st.success("""
            ### ✅ Low Estimated Risk

            The AI model estimates a low likelihood of heart disease.

            **Recommendations**

            • Continue a healthy lifestyle.

            • Maintain regular exercise.

            • Eat a balanced diet.

            • Schedule routine health checkups.
            """)

            elif heart_disease_probability < 0.70:

                st.warning("""
            ### ⚠ Moderate Estimated Risk

            The AI model estimates a moderate likelihood of heart disease.

            **Recommendations**

            • Consult a healthcare professional.

            • Monitor blood pressure and cholesterol.

            • Improve lifestyle habits.

            • Consider additional clinical evaluation.
            """)

            else:

                st.error("""
            ### 🚨 High Estimated Risk

            The AI model estimates a high likelihood of heart disease.

            **Recommendations**

            • Seek medical attention.

            • Schedule a cardiac evaluation.

            • Do not rely solely on AI predictions.

            • Follow physician recommendations.
            """)
            st.divider()

            st.subheader("🧠 AI Interpretation")

            if prediction == 1:

                st.info(f"""
            ### Why did the AI predict Heart Disease?

            The machine learning model estimates a **{heart_disease_probability:.1%}**
            probability of heart disease.

            The SHAP explainability engine identified the clinical features that most
            strongly pushed the prediction toward the **Heart Disease** class.

            This prediction should always be interpreted together with a healthcare
            professional.
            """)

            else:

                st.info(f"""
            ### Why did the AI predict No Heart Disease?

            The machine learning model estimates only a **{heart_disease_probability:.1%}**
            probability of heart disease.

            The SHAP explainability engine found that the patient's clinical features
            were more consistent with the **No Heart Disease** class.

            This prediction should be considered an AI-assisted estimate and not a
            medical diagnosis.
            """)

            # ==========================================
            # CONFIDENCE
            # ==========================================

            with col3:

                st.metric(
                    label="Prediction Confidence",
                    value=f"{predicted_class_probability:.1%}",
                )

                if predicted_class_probability >= 0.90:
                    st.success("Very High")

                elif predicted_class_probability >= 0.70:
                    st.info("Good")

                else:
                    st.warning("Low")
            
            # ======================================
            # PROBABILITY BAR
            # ======================================

            st.subheader(
                "Model-Estimated Heart Disease Probability"
            )

            st.progress(
                heart_disease_probability,
                text=(
                    "Heart Disease Probability: "
                    f"{heart_disease_probability:.1%}"
                ),
            )


            if prediction == 1:

                st.warning(
                    "The machine learning model classified "
                    "this patient's clinical pattern as the "
                    "Heart Disease class."
                )

            else:

                st.success(
                    "The machine learning model classified "
                    "this patient's clinical pattern as the "
                    "No Heart Disease class."
                )


            # ======================================
            # MODEL OUTPUT BAND
            # ======================================

            output_band = (
                get_model_output_band(
                    heart_disease_probability
                )
            )

            st.subheader(
                "Model Output Summary"
            )

            output_col1, output_col2 = (
                st.columns(2)
            )


            with output_col1:

                st.metric(
                    "Model Output Band",
                    output_band,
                )


            with output_col2:

                st.metric(
                    "Estimated Positive-Class Probability",
                    f"{heart_disease_probability:.1%}",
                )


            st.caption(
                "The model output band is a simplified "
                "presentation of the machine-learning "
                "probability. It is not a validated "
                "medical risk category."
            )


            # ======================================
            # SHAP EXPLANATION
            # ======================================

            st.divider()

            st.header(
                "Why Did the AI Make This Prediction?"
            )

            st.write(
                "SHAP values estimate how each transformed "
                "model feature influenced this individual "
                "prediction. Positive values push the model "
                "output toward the Heart Disease class, "
                "while negative values push the output "
                "toward the No Heart Disease class."
            )


            all_contributions = (
                prepare_contribution_dataframe(
                    result.get(
                        "all_contributions"
                    )
                )
            )


            if all_contributions.empty:

                st.warning(
                    "The prediction was generated successfully, "
                    "but no valid SHAP contribution data "
                    "was returned by the explanation engine."
                )


            else:

                # ==================================
                # POSITIVE FACTORS
                # ==================================

                positive_factors = (
                    all_contributions[
                        all_contributions[
                            "shap_value"
                        ] > 0
                    ]
                    .sort_values(
                        "shap_value",
                        ascending=False,
                    )
                    .head(5)
                )


                # ==================================
                # NEGATIVE FACTORS
                # ==================================

                negative_factors = (
                    all_contributions[
                        all_contributions[
                            "shap_value"
                        ] < 0
                    ]
                    .sort_values(
                        "shap_value",
                        ascending=True,
                    )
                    .head(5)
                )


                # ==================================
                # MOST INFLUENTIAL FACTOR
                # ==================================

                st.subheader(
                    "Most Influential Factors"
                )


                strongest_factor = (
                    all_contributions
                    .sort_values(
                        "absolute_shap",
                        ascending=False,
                    )
                    .iloc[0]
                )


                strongest_feature = (
                    strongest_factor[
                        "display_feature"
                    ]
                )


                strongest_shap = (
                    strongest_factor[
                        "shap_value"
                    ]
                )


                if strongest_shap > 0:

                    st.warning(
                        f"The strongest model influence was "
                        f"**{strongest_feature}**, which pushed "
                        f"the prediction toward the "
                        f"Heart Disease class."
                    )

                else:

                    st.success(
                        f"The strongest model influence was "
                        f"**{strongest_feature}**, which pushed "
                        f"the prediction toward the "
                        f"No Heart Disease class."
                    )


                # ==================================
                # POSITIVE / NEGATIVE TABLES
                # ==================================

                col1, col2 = st.columns(2)


                with col1:

                    display_factor_table(
                        positive_factors,
                        "Factors Toward Heart Disease",
                        "positive",
                    )


                with col2:

                    display_factor_table(
                        negative_factors,
                        "Factors Toward No Heart Disease",
                        "negative",
                    )


                # ==================================
                # CONTRIBUTION CHARTS
                # ==================================

                st.divider()

                st.header(
                    "Patient-Level SHAP Contribution Charts"
                )

                col1, col2 = st.columns(2)


                with col1:

                    display_contribution_chart(
                        positive_factors,
                        "Positive Contributions",
                    )


                with col2:

                    display_contribution_chart(
                        negative_factors,
                        "Negative Contributions",
                    )


                # ==================================
                # TOP OVERALL MODEL INFLUENCES
                # ==================================

                st.divider()

                st.subheader(
                    "Top Overall Model Influences"
                )


                top_overall = (
                    all_contributions
                    .sort_values(
                        "absolute_shap",
                        ascending=False,
                    )
                    .head(10)
                    .copy()
                )


                top_overall_display = (
                    top_overall[
                        [
                            "display_feature",
                            "shap_value",
                            "absolute_shap",
                        ]
                    ]
                    .copy()
                )


                top_overall_display.columns = [
                    "Clinical Factor",
                    "SHAP Contribution",
                    "Influence Magnitude",
                ]


                top_overall_display[
                    "SHAP Contribution"
                ] = (
                    top_overall_display[
                        "SHAP Contribution"
                    ]
                    .round(4)
                )


                top_overall_display[
                    "Influence Magnitude"
                ] = (
                    top_overall_display[
                        "Influence Magnitude"
                    ]
                    .round(4)
                )


                st.dataframe(
                    top_overall_display,
                    width="stretch",
                    hide_index=True,
                )


                # ==================================
                # DOWNLOAD PATIENT REPORT
                # ==================================

                st.divider()

                st.header(
                    "Download Patient Analysis Report"
                )


                
                pdf_data = build_patient_pdf(

                    patient_data=
                        patient_data,

                    prediction_label=
                        prediction_label,

                    heart_disease_probability=
                        heart_disease_probability,

                    predicted_confidence=predicted_class_probability,

                    output_band=
                        output_band,

                )
                
                st.write(
                    "Download a text summary containing "
                    "the model prediction, probability, "
                    "patient inputs, SHAP explanation "
                    "factors, and project disclaimer."
                )
                st.download_button(
                    label="📄 Download Analysis Report",

                    data=pdf_data,

                    file_name="bioinsight_ai_patient_report.pdf",

                    mime="application/pdf",

                    width="stretch",

                    type="primary",
                )


            # ======================================
            # PATIENT DATA
            # ======================================

            st.divider()


            with st.expander(
                "View Patient Data Sent to the Model"
            ):

                st.json(
                    patient_data
                )


            # ======================================
            # AI ENGINE OUTPUT DETAILS
            # ======================================

            with st.expander(
                "View AI Engine Output Details"
            ):

                st.write(
                    "Prediction Class:",
                    result[
                        "prediction"
                    ],
                )


                st.write(
                    "Prediction Label:",
                    result[
                        "prediction_label"
                    ],
                )


                st.write(
                    "Positive Class Probability:",
                    result[
                        "positive_class_probability"
                    ],
                )


                st.write(
                    "Predicted Class Confidence:",
                    result[
                        "predicted_class_probability"
                    ],
                )


                st.write(
                    "Number of SHAP Contributions:",
                    len(
                        result[
                            "all_contributions"
                        ]
                    ),
                )


                st.write(
                    "Raw SHAP Contribution Data"
                )


                if not all_contributions.empty:

                    st.dataframe(
                        all_contributions,
                        width="stretch",
                    )


        except Exception as error:

            st.error(
                "An error occurred while analyzing "
                "the patient."
            )

            st.exception(error)


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
---
### 🧬 BioInsight-AI

Developed by **M Ajay Kumar** & **Amogh Amarapur**

**Machine Learning:** Logistic Regression

**Explainability:** SHAP (SHapley Additive Explanations)

**Purpose:** Educational, Research & Demonstration

⚠ This application is **not** intended to replace professional medical diagnosis.

Version **1.0**
"""
)