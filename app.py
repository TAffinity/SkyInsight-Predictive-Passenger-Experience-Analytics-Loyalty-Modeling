import streamlit as st
import pandas as pd
import joblib

# ⚠️ ВАЖНО: только один раз и самым первым
st.set_page_config(page_title="Passenger Loyalty Predictor", layout="centered")

st.title("✈️ Passenger Satisfaction Predictor")
st.write("This AI-powered service predicts whether a passenger will be satisfied with their flight based on service quality metrics.")

# ---------------------------
# MODEL LOAD (только один раз)
# ---------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("airline_rf_pipeline.pkl")

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.write("APP STARTED")

# ---------------------------
# UI
# ---------------------------
st.header("📋 Passenger Survey")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Passenger Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
    age = st.number_input("Passenger Age", 1, 120, 35)

with col2:
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    flight_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    distance = st.number_input("Flight Distance (miles)", 1, 10000, 1000)

# ---------------------------
# Comfort
# ---------------------------
st.subheader("🛋️ InFlight Comfort")

col1, col2 = st.columns(2)

with col1:
    entertainment = st.slider("Inflight entertainment", 1, 5, 3)
    seat_comfort = st.slider("Seat comfort", 1, 5, 3)
    food_drink = st.slider("Food and drink", 1, 5, 3)

with col2:
    cleanliness = st.slider("Cleanliness", 1, 5, 3)
    leg_room_service = st.slider("Leg room service", 1, 5, 3)

# ---------------------------
# Digital
# ---------------------------
st.subheader("💻 Digital Experience")

col1, col2 = st.columns(2)

with col1:
    online_booking_service = st.slider("Ease of online booking", 1, 5, 3)
    online_boarding = st.slider("Online boarding", 1, 5, 3)

with col2:
    wifi_service = st.slider("Inflight wi-fi service", 1, 5, 3)
    online_support = st.slider("Online support", 1, 5, 3)

# ---------------------------
# Airport
# ---------------------------
st.subheader("🧳 Airport & Crew Service")

col1, col2 = st.columns(2)

with col1:
    onboard_service = st.slider("Onboard service", 1, 5, 3)
    checkin_service = st.slider("Check-in service", 1, 5, 3)

with col2:
    baggage_handling = st.slider("Baggage handling", 1, 5, 3)

# ---------------------------
# Reliability
# ---------------------------
st.subheader("⏱️ Flight Reliability & Delays")

col1, col2 = st.columns(2)

with col1:
    gate = st.slider("Gate location convenience", 1, 5, 3)
    dep_val_time_convenient = st.slider("Departure/Arrival time convenient", 1, 5, 3)

with col2:
    departure_delay_minutes = st.number_input("Departure Delay", 0)
    arrival_delay_minutes = st.number_input("Arrival Delay", 0)

# ---------------------------
# INPUT DATA
# ---------------------------
input_data = pd.DataFrame([{
    "gender": gender,
    "customer_type": customer_type,
    "age": age,
    "travel_type": travel_type,
    "class": flight_class,
    "distance": distance,
    "seat_comfort": seat_comfort,
    "dep_val_time_convenient": dep_val_time_convenient,
    "food_drink": food_drink,
    "gate": gate,
    "wifi_service": wifi_service,
    "entertainment": entertainment,
    "online_support": online_support,
    "online_booking_service": online_booking_service,
    "onboard_service": onboard_service,
    "leg_room_service": leg_room_service,
    "baggage_handling": baggage_handling,
    "checkin_service": checkin_service,
    "cleanliness": cleanliness,
    "online_boarding": online_boarding,
    "departure_delay_minutes": departure_delay_minutes,
    "arrival_delay_minutes": arrival_delay_minutes
}])

# dtype fix
input_data["departure_delay_minutes"] = input_data["departure_delay_minutes"].astype("Int64")
input_data["arrival_delay_minutes"] = input_data["arrival_delay_minutes"].astype("Int64")

# feature alignment
try:
    input_data = input_data[pipeline.feature_names_in_]
except:
    pass

# ---------------------------
# PREDICTION
# ---------------------------
st.write("---")

if st.button("📈 Predict Satisfaction", type="primary"):
    prediction = pipeline.predict(input_data)
    proba = pipeline.predict_proba(input_data)

    satisfied_prob = proba[0][1] * 100

    if prediction[0] == 1:
        st.success(f"🎉 SATISFIED (Confidence: {satisfied_prob:.1f}%)")
        st.balloons()
    else:
        st.error(f"😞 DISSATISFIED (Risk: {100 - satisfied_prob:.1f}%)")
