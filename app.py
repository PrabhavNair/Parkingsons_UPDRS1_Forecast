import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile
import matplotlib.pyplot as plt
import os

# -------------------------
# Load model + explainer
# -------------------------
model = xgb.Booster()
model.load_model("models/xgboost_updrs1_model.json")
explainer = shap.TreeExplainer(model)  # not used yet, but fine to keep

# -------------------------
# Sidebar navigation
# -------------------------
page = st.sidebar.selectbox("Select a Page", ["Predictor", "Progression Insight"])

with st.sidebar:
    st.markdown("### ➤ Patient Visit Info")
    st.divider()

    st.markdown("** Month of Clinical Visit**")
    st.markdown("E.g., if this is the 6th month since your first evaluation, enter 6")
    visit_month = st.number_input("Enter month (0–24)", min_value=0, max_value=24, value=0)
    st.divider()

    st.markdown("** UPDRS Part 2: Daily Living Difficulty**")
    st.markdown("""
    How difficult is it for the patient to perform daily activities like:
    - Eating
    - Dressing
    - Maintaining hygiene

    Higher scores = greater difficulty.  
    _(Typical range: 0–52)_
    """)
    updrs_2 = st.number_input("Enter UPDRS Part 2 Score", min_value=0.0, max_value=52.0, value=0.0)
    st.divider()

    st.markdown("** UPDRS Part 3: Motor Examination**")
    st.markdown("""
    This score reflects physical motor symptoms such as:
    - Tremors
    - Muscle rigidity
    - Slowness of movement

    Higher scores = more severe motor impairment.  
    _(Typical range: 0–132)_
    """)
    updrs_3 = st.number_input("Enter UPDRS Part 3 Score", min_value=0.0, max_value=132.0, value=0.0)
    st.divider()

    st.markdown("**💊 UPDRS Part 4: Medication Complications**")
    st.markdown("""
    How severe are medication-related side effects such as:
    - Involuntary movements (dyskinesia)
    - Fluctuations in symptom control
    - 'Wearing off' between doses

    Higher scores = greater complications from medication.  
    _(Typical range: 0–24)_
    """)
    updrs_4 = st.number_input("Enter UPDRS Part 4 Score", min_value=0.0, max_value=24.0, value=0.0)
    st.divider()

# -------------------------
# Expected training columns
# -------------------------
with open("models/updrs1_feature_columns.pkl", "rb") as f:
    expected_columns = joblib.load(f)

def generate_features(month, u2, u3, u4):
    df = pd.DataFrame({
        "visit_month": [month],
        "updrs_2": [u2],
        "updrs_3": [u3],
        "updrs_4": [u4],
        "updrs2_x_updrs3": [u2 * u3],
        "updrs2_x_updrs4": [u2 * u4],
        "updrs3_x_updrs4": [u3 * u4],
        "visit_x_updrs2": [month * u2],
        "visit_x_updrs3": [month * u3],
        "visit_x_updrs4": [month * u4],
        "visit_squared": [month**2],
        "updrs_2_roll3": [u2],
        "updrs_3_roll3": [u3],
        "updrs_1_roll3": [u2 + u3 + u4],
        "interaction_234": [u2 * u3 * u4],
        "updrs_3_squared": [u3**2],
        "updrs_4_squared": [u4**2],
        "log_updrs3": [np.log1p(u3)]
    })
    df = df[expected_columns]
    return df

def get_severity_rubric(score):
    if score < 10:
        return "🟢 Mild symptoms — Monitoring is recommended, but no urgent intervention needed."
    elif score < 17:
        return "🟡 Moderate symptoms — Lifestyle changes and regular follow-ups may be beneficial."
    elif score < 25:
        return "🟠 Severe symptoms likely — Medication adjustments or therapy should be discussed with your neurologist."
    else:
        return "🔴 Very severe symptoms — Immediate clinical intervention and therapy reassessment may be necessary."

# -------------------------
# PDF generator
# -------------------------
def generate_pdf_report(input_data, line_df, recommendation_text):
    """
    input_data: dict with Month, UPDRS_2, UPDRS_3, UPDRS_4, Final_UPDRS_1
    line_df: DataFrame with columns ['visit_month','updrs_1_pred'] for the chart
    """
    # Plot the line using matplotlib (white background)
    plt.figure(figsize=(6.5, 3.5), dpi=150)
    plt.plot(line_df["visit_month"], line_df["updrs_1_pred"], marker='o')
    plt.title("Future UPDRS_1 Predictions")
    plt.xlabel("Month")
    plt.ylabel("Predicted UPDRS_1")
    plt.grid(True, alpha=0.3) 
    plot_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Patient UPDRS Summary", ln=True)

    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Month: {input_data.get('Month', 'N/A')}", ln=True)
    pdf.cell(0, 8, f"UPDRS_2: {input_data.get('UPDRS_2', 'N/A')}", ln=True)
    pdf.cell(0, 8, f"UPDRS_3: {input_data.get('UPDRS_3', 'N/A')}", ln=True)
    pdf.cell(0, 8, f"UPDRS_4: {input_data.get('UPDRS_4', 'N/A')}", ln=True)
    pdf.cell(0, 8, f"Predicted UPDRS_1 after 12 months: {input_data.get('Final_UPDRS_1', 'N/A')}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "UPDRS_1 Future Prediction Chart", ln=True)

    # Insert chart
    pdf.image(plot_path, w=170)

    pdf.ln(4)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Personalized Recommendation", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, recommendation_text)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    try:
        os.remove(plot_path)
    except OSError:
        pass

    return pdf_bytes

# -------------------------
# Pages
# -------------------------
if page == "Predictor":
    st.title("Parkinson's Progression Predictor")

    months_to_simulate = st.sidebar.slider("Months to Simulate", 0, 24, 12)
    future_months = [visit_month + i for i in range(1, months_to_simulate + 1)]

    prediction_df = pd.DataFrame()
    for month in future_months:
        feats = generate_features(month, updrs_2, updrs_3, updrs_4)
        dmatrix = xgb.DMatrix(feats)
        pred = float(model.predict(dmatrix)[0])
        feats["updrs_1_pred"] = pred
        feats["visit_month"] = month
        prediction_df = pd.concat([prediction_df, feats], ignore_index=True)

    final_score = float(prediction_df["updrs_1_pred"].iloc[-1])

    st.markdown(f"### Predicted UPDRS_1 Score after {months_to_simulate} months:")
    st.markdown(
        f"<span style='font-size: 24px; color: lightgreen;'><strong>{final_score:.2f}</strong></span>",
        unsafe_allow_html=True
    )
    st.info(get_severity_rubric(final_score))

    st.markdown("#### What does this mean?")
    st.markdown("""
    - **UPDRS_1** measures non-motor symptoms like cognition, mood, and sleep disturbances.
    - A score above **25** suggests significant mental/cognitive impairment.
    - To help yourself: maintain consistent follow-ups, consider non-pharmacological therapy (CBT, memory support), and review medications with your neurologist.
    """)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prediction_df["visit_month"],
        y=prediction_df["updrs_1_pred"],
        mode='lines+markers',
        marker=dict(size=8),
        hovertemplate='Month %{x}: UPDRS_1 Score %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title="Future UPDRS_1 Predictions",
        xaxis_title="Month",
        yaxis_title="Predicted UPDRS_1",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------- PDF download for the Predictor page (multi-month series) -------
    input_data = {
        "Month": visit_month,
        "UPDRS_2": updrs_2,
        "UPDRS_3": updrs_3,
        "UPDRS_4": updrs_4,
        "Final_UPDRS_1": f"{final_score:.2f}"
    }
    recommendation = "Follow up regularly, maintain lifestyle routines, and seek neurologist input if symptoms worsen."

    # Only pass the two columns needed to draw the line in the PDF
    pdf_series = prediction_df[["visit_month", "updrs_1_pred"]].copy()

    pdf_bytes = generate_pdf_report(
        input_data=input_data,
        line_df=pdf_series,
        recommendation_text=recommendation
    )

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"updrs_report_month_{visit_month}.pdf",
        mime="application/pdf"
    )

elif page == "Progression Insight":
    st.title("Parkinson's Progression Insight")

    feats = generate_features(visit_month, updrs_2, updrs_3, updrs_4)
    dmatrix = xgb.DMatrix(feats)
    pred_score = float(model.predict(dmatrix)[0])

    st.markdown("### Predicted Current UPDRS_1 Score:")
    st.markdown(
        f"<span style='font-size: 24px; color: lightblue;'><strong>{pred_score:.2f}</strong></span>",
        unsafe_allow_html=True
    )
    st.error(get_severity_rubric(pred_score))

    st.markdown("### Clinical Insight:")
    st.markdown("""
    - This predicted score reflects the current severity of non-motor symptoms such as mood, cognition, and sleep.
    - Consistent follow-up is important even in mild stages to track progression.
    - Consider lifestyle changes such as regular exercise, social engagement, and cognitive stimulation.
    - For moderate-to-severe symptoms, coordination with your neurologist is key to managing care.
    """)

    # Optional: single-point PDF (kept if you want it)
    input_data = {
        "Month": visit_month,
        "UPDRS_2": updrs_2,
        "UPDRS_3": updrs_3,
        "UPDRS_4": updrs_4,
        "Final_UPDRS_1": f"{pred_score:.2f}"
    }
    one_point_df = pd.DataFrame({"visit_month": [visit_month], "updrs_1_pred": [pred_score]})
    recommendation = "Follow up regularly, maintain lifestyle routines, and seek neurologist input if symptoms worsen."
    pdf_bytes = generate_pdf_report(input_data, one_point_df, recommendation)

    st.download_button(
        label="Download PDF (current visit)",
        data=pdf_bytes,
        file_name=f"updrs_report_month_{visit_month}.pdf",
        mime="application/pdf"
    )
