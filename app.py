import os
import pickle
from pathlib import Path
from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

from dotenv import load_dotenv
from groq import Groq
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from database import save_prediction, get_all_predictions, search_patient, delete_prediction


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Clinical Decision Support System",
    page_icon="🏥",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

# Always load the .env next to app.py.
load_dotenv(ENV_FILE, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY was not found.\n\n"
        f"Create this file:\n{ENV_FILE}\n\n"
        "and add:\nGROQ_API_KEY=gsk_your_key_here"
    )
    st.stop()

GROQ_API_KEY = GROQ_API_KEY.strip().strip('"').strip("'")

if not GROQ_API_KEY.startswith("gsk_"):
    st.error("The GROQ_API_KEY does not appear to be a valid Groq key.")
    st.stop()

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Could not initialize Groq: {e}")
    st.stop()

GROQ_MODEL = "openai/gpt-oss-20b"


def ask_groq(prompt, max_tokens=300, temperature=0.3):
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI medical information assistant. "
                        "You are not a doctor. Do not provide definitive "
                        "diagnoses or medication changes. Give cautious, "
                        "simple and educational information."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not response.choices:
            raise RuntimeError("Groq returned no choices.")

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError("Groq returned an empty answer.")

        return answer.strip()

    except Exception as e:
        raise RuntimeError(str(e))


# ==========================================================
# OPTIONAL CSS
# ==========================================================

style_file = BASE_DIR / "style.css"
if style_file.exists():
    try:
        st.markdown(
            f"<style>{style_file.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Could not load style.css: {e}")


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    model_file = BASE_DIR / "patient_survival_model.pkl"
    encoder_file = BASE_DIR / "label_encoders.pkl"
    feature_file = BASE_DIR / "feature_columns.pkl"

    missing = [
        p.name for p in [model_file, encoder_file, feature_file]
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError("Missing: " + ", ".join(missing))

    with open(model_file, "rb") as f:
        model = pickle.load(f)
    with open(encoder_file, "rb") as f:
        encoders = pickle.load(f)
    with open(feature_file, "rb") as f:
        feature_columns = pickle.load(f)

    return model, encoders, feature_columns


try:
    model, encoders, feature_columns = load_model()
except Exception as e:
    st.error(f"Unable to load ML files: {e}")
    st.stop()


@st.cache_data
def load_dataset():
    file = BASE_DIR / "support2_dataset.csv"
    if not file.exists():
        raise FileNotFoundError("support2_dataset.csv not found.")
    return pd.read_csv(file)


try:
    df = load_dataset()
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {
    "logged_in": False,
    "page": "Home",
    "prediction_done": False,
    "patient_name": "",
    "patient_data": {},
    "input_df": None,
    "prediction_label": "",
    "survival_probability": 0.0,
    "death_probability": 0.0,
    "dzgroup": "",
    "age": 0,
    "sex": "",
    "chat_history": [],
    "ai_generated": False,
    "ai_error": "",
    "interpretation_text": "",
    "precaution_text": "",
    "diet_plan": "",
    "database_saved_for_prediction": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def safe_classes(name):
    if name not in encoders:
        raise KeyError(f"Encoder '{name}' is missing.")
    return list(encoders[name].classes_)


def reset_prediction():
    for key in [
        "patient_name", "patient_data", "input_df", "prediction_label",
        "dzgroup", "sex", "chat_history", "interpretation_text",
        "precaution_text", "diet_plan", "ai_error"
    ]:
        st.session_state[key] = defaults[key]

    st.session_state.prediction_done = False
    st.session_state.ai_generated = False
    st.session_state.survival_probability = 0.0
    st.session_state.death_probability = 0.0
    st.session_state.age = 0
    st.session_state.database_saved_for_prediction = False


# ==========================================================
# AI GENERATION
# ==========================================================

def generate_ai_outputs(prediction, survival, death, row, disease):
    context = f"""
Prediction: {prediction}
Survival probability: {survival:.2f}%
Death probability: {death:.2f}%
Age: {row.get('age', 'N/A')}
Disease group: {disease}
Mean BP: {row.get('meanbp', 'N/A')}
Heart rate: {row.get('hrt', 'N/A')}
Respiratory rate: {row.get('resp', 'N/A')}
Temperature: {row.get('temp', 'N/A')}
Glucose: {row.get('glucose', 'N/A')}
WBC: {row.get('wblc', 'N/A')}
"""

    interpretation = ask_groq(
        f"""
Patient information:
{context}

Explain the model result and clinical parameters in simple language.
Do not call this a diagnosis. Do not prescribe medicines.
Maximum 200 words.
""",
        300,
        0.2,
    )

    precautions = ask_groq(
        f"""
Patient information:
{context}

Give 8 general monitoring/safety points as bullet points.
Do not prescribe medicines or treatment changes.
Mention professional medical review where appropriate.
""",
        300,
        0.2,
    )

    diet = ask_groq(
        f"""
Patient information:
Age: {row.get('age', 'N/A')}
Disease group: {disease}
Glucose: {row.get('glucose', 'N/A')}
Diabetes: {row.get('diabetes', 'N/A')}

Give a general one-day healthy meal-plan example:
Morning, Breakfast, Lunch, Evening Snack, Dinner, Hydration.
State that an ICU/medically complex patient's diet must be confirmed
by a qualified clinician or dietitian. Do not prescribe a therapeutic diet.
""",
        350,
        0.2,
    )

    return interpretation, precautions, diet


# ==========================================================
# PDF
# ==========================================================

def create_pdf(patient, prediction, survival, death, interpretation, diet, precautions):
    filename = BASE_DIR / "Patient_Report.pdf"

    styles = getSampleStyleSheet()
    title = styles["Heading1"]
    title.alignment = TA_CENTER

    doc = SimpleDocTemplate(
        str(filename),
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )

    table = Table(
        [
            ["Field", "Value"],
            ["Patient Name", str(patient)],
            ["Prediction", str(prediction)],
            ["Survival Probability", f"{survival:.2f}%"],
            ["Death Probability", f"{death:.2f}%"],
        ],
        colWidths=[2.8 * inch, 3.3 * inch],
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B4F8C")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )

    story = [
        Paragraph("AI CLINICAL DECISION SUPPORT SYSTEM", title),
        Spacer(1, 20),
        table,
    ]

    for heading, text in [
        ("AI Clinical Interpretation", interpretation),
        ("AI Diet Recommendation", diet),
        ("AI Precautions", precautions),
    ]:
        story += [
            Spacer(1, 18),
            Paragraph(f"<b>{heading}</b>", styles["Heading2"]),
            Paragraph(escape(str(text)).replace("\n", "<br/>"), styles["BodyText"]),
        ]

    story += [
        Spacer(1, 20),
        Paragraph(
            "Generated: " + datetime.now().strftime("%d-%m-%Y %I:%M %p"),
            styles["BodyText"],
        ),
        Spacer(1, 15),
        Paragraph(
            "This AI-generated report is for educational/decision-support "
            "purposes and should not replace professional medical judgment.",
            styles["BodyText"],
        ),
    ]

    doc.build(story)
    return filename


# ==========================================================
# LOGIN
# ==========================================================

def login_page():
    _, col, _ = st.columns([1, 1.2, 1])

    with col:
        st.title("🏥 AI Clinical Decision Support System")
        st.caption("Secure Hospital Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("🔐 LOGIN", use_container_width=True):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Username or Password")


# ==========================================================
# SIDEBAR
# ==========================================================

def sidebar():
    with st.sidebar:
        st.title("🏥 AI CDSS")

        selected = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "🩺 Predict Patient",
                "📚 Patient History",
                "📊 Analytics",
                "📈 Model Performance",
                "ℹ About",
            ],
            label_visibility="collapsed",
        )

        pages = {
            "🏠 Home": "Home",
            "🩺 Predict Patient": "Predict",
            "📚 Patient History": "History",
            "📊 Analytics": "Analytics",
            "📈 Model Performance": "Performance",
            "ℹ About": "About",
        }

        st.session_state.page = pages[selected]

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()


# ==========================================================
# HOME
# ==========================================================

def home_page():
    st.title("🏥 AI Clinical Decision Support System")
    st.subheader("AI Powered Patient Survival Prediction")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", len(df))
    c2.metric("Features", len(feature_columns))
    c3.metric("Encoders", len(encoders))
    c4.metric("Groq AI", "Connected ✅")

    st.info("Use the sidebar to start a patient prediction.")


# ==========================================================
# PREDICTION FORM
# ==========================================================

def prediction_page():
    st.title("🩺 AI Patient Survival Prediction")

    if st.session_state.prediction_done:
        if st.button("⬅ Back to Patient Form"):
            reset_prediction()
            st.rerun()

        show_prediction_dashboard()
        return

    with st.form("prediction_form"):
        st.subheader("👤 Patient Information")
        c1, c2 = st.columns(2)

        with c1:
            patient_name = st.text_input("Patient Name")
            age = st.number_input("Age", 0, 120, 60)
            sex = st.selectbox("Sex", safe_classes("sex"))
            race = st.selectbox("Race", safe_classes("race"))

        with c2:
            income = st.selectbox("Income", safe_classes("income"))
            edu = st.number_input("Education", value=12)
            num_co = st.number_input("No. of Comorbidities", value=1)

        st.subheader("🩺 Disease Information")
        c1, c2 = st.columns(2)

        with c1:
            dzgroup = st.selectbox("Disease Group", safe_classes("dzgroup"))
            dzclass = st.selectbox("Disease Class", safe_classes("dzclass"))
            ca = st.selectbox("Cancer", safe_classes("ca"))

        with c2:
            diabetes = st.selectbox("Diabetes", [0, 1])
            dementia = st.selectbox("Dementia", [0, 1])
            dnr = st.selectbox("DNR", safe_classes("dnr"))

        st.subheader("❤️ Vital Signs")
        c1, c2, c3 = st.columns(3)

        with c1:
            meanbp = st.number_input("Mean Blood Pressure", value=80.0)
            hrt = st.number_input("Heart Rate", value=90.0)

        with c2:
            resp = st.number_input("Respiratory Rate", value=20.0)
            temp = st.number_input("Temperature", value=37.0)

        with c3:
            wblc = st.number_input("White Blood Cell Count", value=9.0)

        st.subheader("🧪 Laboratory Tests")
        c1, c2, c3 = st.columns(3)

        with c1:
            pafi = st.number_input("PaO₂ / FiO₂", value=300.0)
            alb = st.number_input("Albumin", value=3.5)
            bili = st.number_input("Bilirubin", value=1.0)

        with c2:
            crea = st.number_input("Creatinine", value=1.0)
            sod = st.number_input("Sodium", value=140.0)
            ph = st.number_input("Blood pH", value=7.40, format="%.2f")

        with c3:
            glucose = st.number_input("Glucose", value=120.0)
            bun = st.number_input("Blood Urea Nitrogen", value=18.0)
            urine = st.number_input("Urine Output", value=2500.0)

        st.subheader("📋 Clinical Scores")
        c1, c2 = st.columns(2)

        with c1:
            aps = st.number_input("APS Score", value=20)
            sps = st.number_input("SPS Score", value=15)
            avtisst = st.number_input("APACHE/TISS", value=25)

        with c2:
            scoma = st.number_input("Coma Score", value=0)
            hday = st.number_input("Hospital Days", value=1)
            hospdead = st.selectbox("Hospital Death", [0, 1])

        st.subheader("🧍 Functional Status")
        c1, c2 = st.columns(2)

        with c1:
            adlp = st.number_input("ADL Physical", value=5)
            adls = st.number_input("ADL Score", value=5)

        with c2:
            adlsc = st.number_input("ADL Change Score", value=5)
            sfdm2 = st.selectbox("Functional Status", safe_classes("sfdm2"))

        submit = st.form_submit_button(
            "🚀 Predict Patient Survival",
            use_container_width=True,
        )

    if not submit:
        return

    try:
        data = {
            "age": age, "sex": sex, "dzgroup": dzgroup, "dzclass": dzclass,
            "num.co": num_co, "edu": edu, "income": income, "scoma": scoma,
            "avtisst": avtisst, "race": race, "sps": sps, "aps": aps,
            "hday": hday, "diabetes": diabetes, "dementia": dementia,
            "ca": ca, "dnr": dnr, "meanbp": meanbp, "wblc": wblc,
            "hrt": hrt, "resp": resp, "temp": temp, "pafi": pafi,
            "alb": alb, "bili": bili, "crea": crea, "sod": sod, "ph": ph,
            "glucose": glucose, "bun": bun, "urine": urine, "adlp": adlp,
            "adls": adls, "adlsc": adlsc, "hospdead": hospdead, "sfdm2": sfdm2,
        }

        categorical = [
            "sex", "dzgroup", "dzclass", "income",
            "race", "ca", "dnr", "sfdm2"
        ]

        for col in categorical:
            if col not in encoders:
                raise KeyError(f"Encoder for '{col}' is missing.")
            data[col] = encoders[col].transform([data[col]])[0]

        input_df = pd.DataFrame([data])

        missing = [x for x in feature_columns if x not in input_df.columns]
        if missing:
            raise ValueError("Missing model features: " + ", ".join(missing))

        input_df = input_df[feature_columns]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        death = round(float(probability[0]) * 100, 2)
        survival = round(float(probability[1]) * 100, 2)

        st.session_state.prediction_done = True
        st.session_state.patient_name = patient_name
        st.session_state.patient_data = data
        st.session_state.input_df = input_df
        st.session_state.prediction_label = (
            "Survived" if prediction == 1 else "Not Survived"
        )
        st.session_state.survival_probability = survival
        st.session_state.death_probability = death
        st.session_state.dzgroup = dzgroup
        st.session_state.age = age
        st.session_state.sex = sex
        st.session_state.ai_generated = False
        st.session_state.chat_history = []
        st.session_state.ai_error = ""
        st.session_state.database_saved_for_prediction = False

        st.rerun()

    except Exception as e:
        st.error(f"Prediction Error: {e}")


# ==========================================================
# DASHBOARD
# ==========================================================

def show_prediction_dashboard():
    survival = float(st.session_state.survival_probability)
    death = float(st.session_state.death_probability)
    prediction = st.session_state.prediction_label
    patient = st.session_state.patient_name
    disease = st.session_state.dzgroup
    row = st.session_state.input_df.iloc[0]

    meanbp = float(row["meanbp"])
    hrt = float(row["hrt"])
    resp = float(row["resp"])
    temp = float(row["temp"])
    wblc = float(row["wblc"])
    glucose = float(row["glucose"])

    st.header("📊 Clinical Prediction Dashboard")

    if survival >= 80:
        risk = "🟢 LOW RISK"
    elif survival >= 50:
        risk = "🟡 MODERATE RISK"
    else:
        risk = "🔴 HIGH RISK"

    st.info(
        f"**Patient:** {patient}  \n"
        f"**Prediction:** {prediction}  \n"
        f"**Risk:** {risk}"
    )

    c1, c2 = st.columns(2)

    with c1:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=survival,
                number={"suffix": "%"},
                title={"text": "Survival Probability"},
                gauge={"axis": {"range": [0, 100]}},
            )
        )
        st.plotly_chart(gauge, use_container_width=True)

    with c2:
        pie = px.pie(
            values=[survival, death],
            names=["Survival", "Death"],
            hole=0.5,
        )
        st.plotly_chart(pie, use_container_width=True)

    a, b, c = st.columns(3)
    a.metric("Prediction", prediction)
    b.metric("Survival", f"{survival:.2f}%")
    c.metric("Death", f"{death:.2f}%")

    st.subheader("❤️ Vital Signs")
    a, b, c = st.columns(3)
    a.metric("Mean BP", meanbp)
    a.metric("Heart Rate", hrt)
    b.metric("Respiratory Rate", resp)
    b.metric("Temperature", temp)
    c.metric("WBC", wblc)
    c.metric("Glucose", glucose)

    # SHAP
    st.subheader("🧠 SHAP Explainability")
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(st.session_state.input_df)

        if isinstance(values, list):
            values = values[-1]
        if hasattr(values, "ndim") and values.ndim == 3:
            values = values[:, :, -1]

        fig_shap = plt.figure(figsize=(10, 4))
        shap.summary_plot(
            values,
            st.session_state.input_df,
            plot_type="bar",
            max_display=10,
            show=False,
        )
        st.pyplot(fig_shap)
        plt.close(fig_shap)
    except Exception as e:
        st.warning(f"SHAP Error: {e}")

    # AI output
    st.divider()
    st.subheader("🤖 AI Clinical Support")

    if st.button(
        "✨ Generate AI Clinical Interpretation, Precautions & Diet",
        disabled=st.session_state.ai_generated,
        use_container_width=True,
    ):
        try:
            with st.spinner("Generating response using Groq..."):
                interpretation, precautions, diet = generate_ai_outputs(
                    prediction, survival, death, row, disease
                )

            st.session_state.interpretation_text = interpretation
            st.session_state.precaution_text = precautions
            st.session_state.diet_plan = diet
            st.session_state.ai_generated = True
            st.rerun()

        except Exception as e:
            st.error(f"Groq API Error: {e}")

    if st.session_state.ai_generated:
        st.info(st.session_state.interpretation_text)
        st.warning(st.session_state.precaution_text)
        st.success(st.session_state.diet_plan)

        try:
            pdf = create_pdf(
                patient,
                prediction,
                survival,
                death,
                st.session_state.interpretation_text,
                st.session_state.diet_plan,
                st.session_state.precaution_text,
            )
            with open(pdf, "rb") as f:
                st.download_button(
                    "📄 Download Medical Report",
                    f,
                    "Patient_Report.pdf",
                    "application/pdf",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"PDF generation error: {e}")

    # Chatbot
    st.divider()
    st.subheader("💬 AI Medical Assistant")

    question = st.text_input(
        "Ask AI...",
        placeholder="Example: Why is survival probability low?",
        key="medical_chat_question",
    )

    if st.button("🤖 Ask AI", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            context = f"""
Patient: {patient}
Prediction: {prediction}
Survival Probability: {survival:.2f}%
Death Probability: {death:.2f}%
Disease Group: {disease}
Age: {st.session_state.age}
Mean BP: {meanbp}
Heart Rate: {hrt}
Respiratory Rate: {resp}
Temperature: {temp}
Glucose: {glucose}
WBC: {wblc}
"""

            prompt = f"""
Patient information:
{context}

Question:
{question}

Answer in simple medical language in under 180 words.
Do not diagnose the patient, prescribe medicines, or claim to be a doctor.
Explain when professional medical evaluation may be needed.
"""

            try:
                with st.spinner("Generating response using Groq..."):
                    answer = ask_groq(prompt, 250, 0.3)

                st.session_state.chat_history.append(("You", question))
                st.session_state.chat_history.append(("AI", answer))

            except Exception as e:
                st.error(f"Groq API Error: {e}")

    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(
                f"<div style='background:#2563EB;padding:12px;border-radius:10px;"
                f"margin-bottom:10px;color:white'><b>🧑 You</b><br>"
                f"{escape(str(message))}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:white;padding:12px;border-radius:10px;"
                f"margin-bottom:15px;color:black;border-left:5px solid #22C55E'>"
                f"<b>🤖 AI Medical Assistant</b><br>{escape(str(message))}</div>",
                unsafe_allow_html=True,
            )

    # Alerts
    st.divider()
    st.subheader("🚨 Clinical Alert Checks")

    alerts = False
    if meanbp < 60:
        st.error("Low blood pressure detected.")
        alerts = True
    if glucose > 250:
        st.error("High blood glucose detected.")
        alerts = True
    if temp > 39:
        st.error("High temperature detected.")
        alerts = True
    if resp > 30:
        st.error("High respiratory rate detected.")
        alerts = True
    if wblc > 15:
        st.warning("Elevated WBC count detected.")
        alerts = True

    if not alerts:
        st.success("No simple alert threshold was triggered.")

    # Health score
    health = 100
    if meanbp < 65: health -= 15
    if glucose > 180: health -= 10
    if temp > 38: health -= 10
    if resp > 25: health -= 10
    if wblc > 11: health -= 10
    if survival < 70: health -= 20
    health = max(0, health)

    st.subheader("🩺 Overall Health Score")
    st.progress(health / 100)
    st.metric("Health Score", f"{health}/100")

    # Clinical risk meter
    risk_data = {
        "Blood Pressure": 100 if meanbp >= 65 else 35,
        "Temperature": 100 if temp <= 38 else 40,
        "Respiration": 100 if resp <= 25 else 45,
        "Glucose": 100 if glucose <= 180 else 55,
        "WBC": 100 if wblc <= 11 else 60,
    }

    risk_df = pd.DataFrame(
        {"Parameter": risk_data.keys(), "Score": risk_data.values()}
    )

    fig = px.bar(
        risk_df,
        x="Parameter",
        y="Score",
        text="Score",
        range_y=[0, 100],
        title="Clinical Stability",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Database
    if not st.session_state.database_saved_for_prediction:
        try:
            save_prediction(
                patient_name=patient,
                age=int(st.session_state.age),
                gender=str(st.session_state.sex),
                disease_group=str(disease),
                survival_probability=survival,
                death_probability=death,
                prediction=prediction,
            )
            st.session_state.database_saved_for_prediction = True
        except Exception as e:
            st.warning(f"Database save error: {e}")


# ==========================================================
# HISTORY
# ==========================================================

def history_page():
    st.title("📚 Patient History")

    try:
        history = get_all_predictions()
    except Exception as e:
        st.error(f"Unable to load history: {e}")
        return

    search = st.text_input("🔍 Search Patient")

    if search.strip():
        try:
            history = search_patient(search)
        except Exception as e:
            st.error(f"Search error: {e}")
            return

    if history is None or history.empty:
        st.info("No patient history available.")
        return

    st.dataframe(history, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download History",
        history.to_csv(index=False).encode(),
        "patient_history.csv",
        "text/csv",
    )

    st.subheader("🗑 Delete Record")
    record_id = st.number_input("Record ID", min_value=1, step=1)

    if st.button("Delete Record"):
        try:
            delete_prediction(record_id)
            st.success("Record deleted.")
            st.rerun()
        except Exception as e:
            st.error(f"Delete error: {e}")


# ==========================================================
# ANALYTICS
# ==========================================================

def analytics_page():
    st.title("📊 Analytics Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(df))
    c2.metric("Features", len(df.columns))
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    if "age" in df.columns:
        fig = px.histogram(df, x="age", nbins=30, title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if "hospdead" in df.columns:
        fig = px.pie(
            df,
            names="hospdead",
            title="Hospital Death Distribution",
            hole=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.head(20), use_container_width=True)


# ==========================================================
# PERFORMANCE
# ==========================================================

def performance_page():
    st.title("📈 Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "92.8%")
    c2.metric("Precision", "91.9%")
    c3.metric("Recall", "93.5%")
    c4.metric("F1 Score", "92.7%")

    data = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [92.8, 91.9, 93.5, 92.7],
    })

    fig = px.bar(data, x="Metric", y="Score", text="Score")
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# ABOUT
# ==========================================================

def about_page():
    st.title("ℹ About Project")

    st.write("""
This AI Clinical Decision Support System predicts patient survival
using a machine-learning model trained on the SUPPORT2 dataset.

It includes ML prediction, SHAP explainability, Groq-powered AI
information, patient history, analytics and PDF reporting.

This is an educational/research prototype and must not replace
professional medical judgment.
""")

    st.write("""
**Technologies:** Streamlit, Scikit-learn, Pandas, NumPy,
Plotly, Matplotlib, SHAP, Groq API and MySQL.
""")


# ==========================================================
# MAIN
# ==========================================================

if not st.session_state.logged_in:
    login_page()
else:
    sidebar()

    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Predict":
        prediction_page()
    elif st.session_state.page == "History":
        history_page()
    elif st.session_state.page == "Analytics":
        analytics_page()
    elif st.session_state.page == "Performance":
        performance_page()
    elif st.session_state.page == "About":
        about_page()