import streamlit as st
import joblib
import pandas as pd

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Dry Eye Screen Health Assistant",
    page_icon="👁️👁️💧",
    layout="centered"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("👁️👁️💧 Dry Eye & Screen Health Assistant")

st.write(
    """
    This application uses a machine learning model to estimate
    the likelihood of Dry Eye Disease based on lifestyle,
    sleep and digital screen-use factors.
    """
)

st.info(
    "⚠️ This is an experimental machine-learning tool "
    "and is NOT a medical diagnosis."
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
try:
    model = joblib.load("lesley.joblib")
except Exception as e:
    st.error("Could not load the model due to a version mismatch.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# USER INFORMATION
# --------------------------------------------------
st.header("🧑 About You")

gender_select = st.selectbox("Gender", ["Female", "Male"])
gender = 1 if gender_select == "Female" else 0

age = st.number_input("Age", min_value=18, max_value=100, value=25)

# --------------------------------------------------
# BLOOD PRESSURE
# --------------------------------------------------
st.header("❤️ Cardiovascular Health")

systolic_bp = st.number_input(
    "Systolic Blood Pressure (mmHg)", 
    min_value=70, max_value=200, value=120
)

diastolic_bp = st.number_input(
    "Diastolic Blood Pressure (mmHg)", 
    min_value=40, max_value=130, value=80
)

# --------------------------------------------------
# SCREEN EXPOSURE
# --------------------------------------------------
st.header("📱 Your Screen Habits")

screen_time = st.number_input(
    "Average screen time (hours per day)",
    min_value=0.0, max_value=24.0, value=6.0, step=0.1
)

smart_device_select = st.selectbox("Do you use a smart device before bed?", ["No", "Yes"])
smart_device = 1 if smart_device_select == "Yes" else 0

blue_light_select = st.selectbox("Do you use a blue-light filter?", ["No", "Yes"])
blue_light = 1 if blue_light_select == "Yes" else 0

# --------------------------------------------------
# SLEEP
# --------------------------------------------------
st.header("😴 Your Sleep")

sleep_duration = st.number_input(
    "Sleep duration (hours per night)",
    min_value=0.0, max_value=24.0, value=7.0, step=0.1
)
sleep_quality = st.slider("Sleep quality", min_value=1, max_value=5, value=3)

sleep_disorder_select = st.selectbox("Do you have a sleep disorder?", ["No", "Yes"])
sleep_disorder = 1 if sleep_disorder_select == "Yes" else 0

wake_up_select = st.selectbox("Do you wake up during the night?", ["No", "Yes"])
wake_up = 1 if wake_up_select == "Yes" else 0

sleepy_day_select = st.selectbox("Do you feel sleepy during the day?", ["No", "Yes"])
sleepy_day = 1 if sleepy_day_select == "Yes" else 0

# --------------------------------------------------
# STRESS AND PHYSICAL ACTIVITY
# --------------------------------------------------
st.header("🏃 Lifestyle")

stress = st.slider("Stress level", min_value=1, max_value=5, value=3)
physical_activity = st.number_input("Physical activity (minutes per day)", min_value=0, max_value=500, value=30)
daily_steps = st.number_input("Daily steps", min_value=0, max_value=50000, value=5000)

# --------------------------------------------------
# OTHER LIFESTYLE FACTORS
# --------------------------------------------------
caffeine_select = st.selectbox("Caffeine consumption", ["No", "Yes"])
caffeine = 1 if caffeine_select == "Yes" else 0

alcohol_select = st.selectbox("Alcohol consumption", ["No", "Yes"])
alcohol = 1 if alcohol_select == "Yes" else 0

smoking_select = st.selectbox("Smoking", ["No", "Yes"])
smoking = 1 if smoking_select == "Yes" else 0

medical_issue_select = st.selectbox("Do you have a medical issue?", ["No", "Yes"])
medical_issue = 1 if medical_issue_select == "Yes" else 0

medication_select = st.selectbox("Are you taking ongoing medication?", ["No", "Yes"])
medication = 1 if medication_select == "Yes" else 0

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------
if st.button("🔍 Check My Dry Eye Risk", use_container_width=True):
    try:
        # ------------------------------------------
        # CREATE INPUT DATAFRAME (All numeric values now)
        # ------------------------------------------
        input_data = pd.DataFrame({
            "Gender": [gender],
            "Age": [age],
            "Sleep duration": [sleep_duration],
            "Sleep quality": [sleep_quality],
            "Stress level": [stress],
            "Daily steps": [daily_steps],
            "Physical activity": [physical_activity],
            "Sleep disorder": [sleep_disorder],
            "Wake up during night": [wake_up],
            "Feel sleepy during day": [sleepy_day],
            "Caffeine consumption": [caffeine],
            "Alcohol consumption": [alcohol],
            "Smoking": [smoking],
            "Medical issue": [medical_issue],
            "Ongoing medication": [medication],
            "Smart device before bed": [smart_device],
            "Average screen time": [screen_time],
            "Blue-light filter": [blue_light],
            "Systolic_BP": [systolic_bp],       
            "Diastolic_BP": [diastolic_bp]       
        })

        # ------------------------------------------
        # MAKE PREDICTION
        # ------------------------------------------
        prediction = model.predict(input_data)
        
        # Extract binary probability float safely
        probability = float(model.predict_proba(input_data)[0][1])
        risk_percentage = probability * 100

        # ------------------------------------------
        # DISPLAY RESULT
        # ------------------------------------------
        st.header("📊 Your Result")
        st.metric("Estimated Dry Eye Risk", f"{risk_percentage:.1f}%")

        # ------------------------------------------
        # RISK MESSAGE
        # ------------------------------------------
        if probability >= 0.70:
            st.error("Your estimated risk is relatively high.")
        elif probability >= 0.40:
            st.warning("Your estimated risk is moderate.")
        else:
            st.success("Your estimated risk is relatively low.")

        # ------------------------------------------
        # SCREEN TIME ADVICE
        # ------------------------------------------
        st.header("💡 Personalized Screen Health Advice")
        if screen_time >= 8:
            st.warning("Your screen time is high. Consider using the 20-20-20 rule to rest your eyes.")
        else:
            st.success("Your daily screen exposure is within a manageable range.")

    except Exception as prediction_error:
        st.error("Prediction failed. Ensure features exactly match your training data layout.")
        st.exception(prediction_error)
