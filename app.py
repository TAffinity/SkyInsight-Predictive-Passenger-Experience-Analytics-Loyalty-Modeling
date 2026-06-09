import streamlit as st
import pandas as pd
import joblib

st.write("APP STARTED")
try:
    model = joblib.load("airline_rf_pipeline.pkl")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# 1. Page Configuration
st.set_page_config(page_title="Passenger Loyalty Predictor", layout="centered")
st.title("✈️ Passenger Satisfaction Predictor")
# st.markdown("### 🎯 **Model Predictive Accuracy: 96%** | **Model Reliability: 99%**")
st.write("This AI-powered service predicts whether a passenger will be satisfied with their flight based on service quality metrics.")

# 2. Pipeline Loading Function
@st.cache_resource
def load_pipeline():
    return joblib.load('airline_rf_pipeline.pkl')
try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Failed to load the model file 'airline_rf_pipeline.pkl'. Error: {e}")
    st.stop()

# 3. Passenger Survey User Interface
st.header("📋 Passenger Survey")

# Demographics and Flight Context (Basic Info)
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Passenger Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
    age = st.number_input("Passenger Age", min_value=1, max_value=120, value=35)
with col2:
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    flight_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"], index=0)
    distance = st.number_input("Flight Distance (miles)", min_value=1, max_value=10000, value=1000)

# --- CATEGORY 1: InFlight Comfort ---
st.subheader("🛋️ InFlight Comfort (Rating: 1 to 5)")
col_inf1, col_inf2 = st.columns(2)
with col_inf1:
    entertainment = st.slider("Inflight entertainment", 1, 5, 3)
    seat_comfort = st.slider("Seat comfort", 1, 5, 3)
    food_drink = st.slider("Food and drink", 1, 5, 3)
with col_inf2:
    cleanliness = st.slider("Cleanliness", 1, 5, 3)
    leg_room_service = st.slider("Leg room service", 1, 5, 3)

# --- CATEGORY 2: Digital Experience ---
st.subheader("💻 Digital Experience (Rating: 1 to 5)")
col_dig1, col_dig2 = st.columns(2)
with col_dig1:
    online_booking_service = st.slider("Ease of online booking", 1, 5, 3)
    online_boarding = st.slider("Online boarding", 1, 5, 3)
with col_dig2:
    wifi_service = st.slider("Inflight wi-fi service", 1, 5, 3)
    online_support = st.slider("Online support", 1, 5, 3)

# --- CATEGORY 3: Airport & Crew Service ---
st.subheader("🧳 Airport & Crew Service (Rating: 1 to 5)")
col_air1, col_air2 = st.columns(2)
with col_air1:
    onboard_service = st.slider("Onboard service", 1, 5, 3)
    checkin_service = st.slider("Check-in service", 1, 5, 3)
with col_air2:
    baggage_handling = st.slider("Baggage handling", 1, 5, 3)

# --- CATEGORY 4: Flight Reliability ---
st.subheader("⏱️ Flight Reliability & Delays")
col_rel1, col_rel2 = st.columns(2)
with col_rel1:
    gate = st.slider("Gate location convenience (1 to 5)", 1, 5, 3)
    dep_val_time_convenient = st.slider("Departure/Arrival time convenient (1 to 5)", 1, 5, 3)
with col_rel2:
    departure_delay_minutes = st.number_input("Departure Delay (minutes)", min_value=0, value=0)
    arrival_delay_minutes = st.number_input("Arrival Delay (minutes)", min_value=0, value=0)

# 4. Dataframe Construction for Prediction
input_data = pd.DataFrame([{
    'gender': gender,
    'customer_type': customer_type,
    'age': age,
    'travel_type': travel_type,
    'class': flight_class, # Оставьте с большой 'Class', если модель ждет ее так
    'distance': distance,
    'seat_comfort': seat_comfort,
    'dep_val_time_convenient': dep_val_time_convenient,
    'food_drink': food_drink,
    'gate': gate,
    'wifi_service': wifi_service,
    'entertainment': entertainment,
    'online_support': online_support,
    'online_booking_service': online_booking_service,
    'onboard_service': onboard_service,
    'leg_room_service': leg_room_service,
    'baggage_handling': baggage_handling,
    'checkin_service': checkin_service,
    'cleanliness': cleanliness,
    'online_boarding': online_boarding,
    'departure_delay_minutes': departure_delay_minutes,
    'arrival_delay_minutes': arrival_delay_minutes
}])
# Aligning data types with the trained pipeline
input_data['departure_delay_minutes'] = input_data['departure_delay_minutes'].astype('Int64')
input_data['arrival_delay_minutes'] = input_data['arrival_delay_minutes'].astype('Int64')

# Strictly matching the feature order expected by the pipeline
try:
    expected_cols = pipeline.feature_names_in_
    input_data = input_data[expected_cols]
except:
    pass

# 5. Prediction Execution
st.write("---")
if st.button("📈 Predict Satisfaction", type="primary"):
    prediction = pipeline.predict(input_data)
    probabilities = pipeline.predict_proba(input_data)
    satisfied_prob = probabilities[0][1] * 100

    if prediction[0] == 1:
        st.success(f"🎉 **The passenger will be SATISFIED!** (Retention Confidence: {satisfied_prob:.1f}%)")
        st.balloons()
    else:
        st.error(f"😞 **The passenger will be DISSATISFIED** (Churn Risk Level: {100 - satisfied_prob:.1f}%)")
