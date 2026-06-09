import streamlit as st
import pandas as pd
import joblib

# ✅ MUST BE FIRST STREAMLIT CALL
st.set_page_config(
    page_title="Passenger Loyalty Predictor",
    layout="centered"
)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("airline_rf_pipeline.pkl")

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -------------------------
# HEADER
# -------------------------
st.title("✈️ Passenger Satisfaction Predictor")
st.write(
    "This AI-powered service predicts whether a passenger will be satisfied "
    "based on service quality metrics."
)

st.write("APP STARTED")

# -------------------------
# USER INPUTS
# -------------------------
st.header("📋 Passenger Survey")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Passenger Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
    age = st.number_input("Passenger Age", min_value=1, max_value=120, value=35)

with col2:
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    flight_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"], index=0)
    distance = st.number_input("Flight Distance (miles)", min_value=1, max_value=10000, value=1000)

st.subheader("🛋️ InFlight Comfort")

entertainment = st.slider("Inflight entertainment", 1, 5, 3)
seat_comfort = st.slider("Seat comfort", 1, 5, 3)
food_drink = st.slider("Food and drink", 1, 5, 3)
cleanliness = st.slider("Cleanliness", 1, 5, 3)
leg_room_service = st.slider("Leg room service", 1, 5, 3)

st.subheader("💻 Digital Experience")

online_booking_service = st.slider("Ease of online booking", 1, 5, 3)
online_boarding = st.slider("Online boarding", 1, 5, 3)
wifi_service = st.slider("Inflight Wi-Fi service", 1, 5, 3)
online_support = st.slider("Online support", 1, 5, 3)

st.subheader("🧳 Airport & Crew Service")

onboard_service = st.slider("Onboard service", 1, 5, 3)
checkin_service = st.slider("Check-in service", 1, 5, 3)
baggage_handling = st.slider("Baggage handling", 1, 5, 3)

st.subheader("⏱️ Flight Reliability")

gate = st.slider("Gate location convenience", 1, 5, 3)
dep_val_time_convenient = st.slider("Departure/Arrival time convenient", 1, 5, 3)
departure_delay_minutes = st.number_input("Departure Delay (minutes)", min_value=0, value=0)
arrival_delay_minutes = st.number_input("Arrival Delay (minutes)", min_value=0, value=0)

# -------------------------
# BUILD INPUT DATA
# -------------------------
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

# match types
input_data["departure_delay_minutes"] = input_data["departure_delay_minutes"].astype("Int64")
input_data["arrival_delay_minutes"] = input_data["arrival_delay_minutes"].astype("Int64")

# align columns if model supports it
try:
    input_data = input_data[pipeline.feature_names_in_]
except:
    pass

# -------------------------
# PREDICTION
# -------------------------
st.write("---")

if st.button("📈 Predict Satisfaction", type="primary"):
    prediction = pipeline.predict(input_data)
    proba = pipeline.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success(f"🎉 SATISFIED (Confidence: {proba:.1%})")
        st.balloons()
    else:
        st.error(f"😞 DISSATISFIED (Risk: {1 - proba:.1%})")
