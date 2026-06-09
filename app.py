import joblib
import pandas as pd
import streamlit as st
# import plotly.graph_objects as go

# 1. Page Configuration & Custom Executive Theme
st.set_page_config(page_title="SkyInsight Executive Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stSlider > label { font-weight: 600; color: #2c3e50; font-size: 14px; }
    .category-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
        border-left: 6px solid #ccc;
    }
    .comfort-box { border-left-color: #99ff99; }
    .digital-box { border-left-color: #66b3ff; }
    .service-box { border-left-color: #ff9999; }
    .reliability-box { border-left-color: #ffb84d; }
    .result-card {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 22px;
        color: white;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .success-card { background: linear-gradient(135deg, #2ecc71, #27ae60); box-shadow: 0 6px 15px rgba(46, 204, 113, 0.3); }
    .error-card { background: linear-gradient(135deg, #e74c3c, #c0392b); box-shadow: 0 6px 15px rgba(231, 76, 60, 0.3); }
    .action-box {
        background-color: #fff5f5;
        border: 1px dashed #e74c3c;
        padding: 20px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .rec-item {
        margin-bottom: 8px;
        font-size: 14px;
        line-height: 1.4;
    }
    [data-testid="stMetricValue"] {
        font-size: 15px !important;
    }

    </style>
""", unsafe_allow_html=True)

# 2. Pipeline Loading Function
@st.cache_resource
def load_pipeline():
    return joblib.load('airline_rf_pipeline.pkl')
try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Failed to load the model file 'airline_rf_pipeline.pkl'. Error: {e}")
    st.stop()

# --- MAIN CONTENT AREA ---
st.title("✈️ SkyInsight: Predictive Passenger Experience Analytics")
st.write("Simulate service performance metrics below to compute operational retention forecasting in real-time.")

# 3. High-Level Executive KPIs
st.markdown("### 🎯 System Diagnostic Performance")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1: st.metric(label="Overall Accuracy", value="96%")
with kpi2: st.metric(label="Model Reliability (AUC)", value="99%")
with kpi3: st.metric(label="Prediction Precision", value="97%")
with kpi4: st.metric(label="Risk Detection (Recall)", value="96%")
st.markdown("---")

# 3. Passenger Survey User Interface
st.header("📋 Passenger Survey")
# Demographics and Flight Context (Basic Info)
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Passenger Gender", ["Male", "Female"])
    age = st.number_input("Passenger Age", min_value=1, max_value=120, value=40)
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"], index=0)
with col2:
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    flight_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"], index=0)
    distance = st.number_input("Flight Distance (miles)", min_value=1, max_value=15000, value=1800)

left_layout, right_layout = st.columns(2, gap="large")
with left_layout:
    st.markdown("### 🛠️ Service Touchpoint Simulator")
    # CATEGORY 1: InFlight Comfort
    st.markdown('<div class="category-box comfort-box">', unsafe_allow_html=True)
    st.markdown("#### 🛋️ InFlight Comfort")
    c1, c2 = st.columns(2)
    with c1:
        entertainment = st.slider("Inflight entertainment", 1, 5, 3)
        seat_comfort = st.slider("Seat comfort", 1, 5, 3)
        food_drink = st.slider("Food and drink", 1, 5, 3)
    with c2:
        cleanliness = st.slider("Cleanliness", 1, 5, 3)
        leg_room_service = st.slider("Leg room service", 1, 5, 3)
    st.markdown('</div>', unsafe_allow_html=True)
    # CATEGORY 2: Digital Experience
    st.markdown('<div class="category-box digital-box">', unsafe_allow_html=True)
    st.markdown("#### 💻 Digital Experience")
    d1, d2 = st.columns(2)
    with d1:
        online_booking_service = st.slider("Ease of online booking", 1, 5, 3)
        online_boarding = st.slider("Online boarding", 1, 5, 3)
    with d2:
        wifi_service = st.slider("Inflight wi-fi service", 1, 5, 3)
        online_support = st.slider("Online support", 1, 5, 3)
    st.markdown('</div>', unsafe_allow_html=True)
    # CATEGORY 3: Airport & Crew Service
    st.markdown('<div class="category-box service-box">', unsafe_allow_html=True)
    st.markdown("#### 🧳 Airport & Crew Service")
    a1, a2 = st.columns(2)
    with a1:
        onboard_service = st.slider("Onboard service", 1, 5, 3)
        checkin_service = st.slider("Check-in service", 1, 5, 3)
    with a2:
        baggage_handling = st.slider("Baggage handling", 1, 5, 3)
    st.markdown('</div>', unsafe_allow_html=True)
    # CATEGORY 4: Flight Reliability
    st.markdown('<div class="category-box reliability-box">', unsafe_allow_html=True)
    st.markdown("#### ⏱️ Flight Reliability & Delays")
    r1, r2 = st.columns(2)
    with r1:
        gate = st.slider("Gate location convenience", 1, 5, 3)
        dep_val_time_convenient = st.slider("Time convenience", 1, 5, 3)
    with r2:
        departure_delay_minutes = st.number_input("Departure Delay (min)", min_value=0, value=0)
        arrival_delay_minutes = st.number_input("Arrival Delay (min)", min_value=0, value=0)
    st.markdown('</div>', unsafe_allow_html=True)
with right_layout:
    st.markdown("### 📊 Experience Geometry & Scoring")
    comfort_avg = (entertainment + seat_comfort + food_drink + cleanliness + leg_room_service) / 5
    digital_avg = (online_booking_service + online_boarding + wifi_service + online_support) / 4
    service_avg = (onboard_service + checkin_service + baggage_handling) / 3
    reliability_avg = (gate + dep_val_time_convenient) / 2

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[comfort_avg, digital_avg, service_avg, reliability_avg, comfort_avg],
        theta=['InFlight Comfort', 'Digital Experience', 'Airport & Crew', 'Flight Reliability', 'InFlight Comfort'],
        fill='toself',
        fillcolor='rgba(102, 179, 255, 0.2)',
        line=dict(color='#66b3ff', width=3),
        name='Current Service Profile'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=340)
    st.plotly_chart(fig, use_container_width=True)

    # 4. Dataframe Construction for Prediction
    input_data = pd.DataFrame([{
        'gender': gender,
        'customer_type': customer_type,  # Перенесено на позицию 2
        'age': age,
        'travel_type': travel_type,
        'class': flight_class,
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
        # 'departure_delay_minutes': int(departure_delay_minutes),
        'arrival_delay_minutes': float(arrival_delay_minutes)
    }])
##
    try:
        expected_cols = pipeline.feature_names_in_
        input_data = input_data[expected_cols]
    except:
        pass
    # 5. Prediction & Custom Threshold Execution (Панель перенесена направо)
    st.markdown("### 📈 Risk Evaluation Execution")
    retention_threshold = st.slider(
        "Minimum Satisfaction Threshold (%)",
        min_value=50, max_value=90, value=65, step=5,
        help="The minimum probability required to consider a passenger safely retained.")
    if st.button("Execute Strategic Predictive Scoring", type="primary", use_container_width=True):
        probabilities = pipeline.predict_proba(input_data)
        satisfied_prob = probabilities[0][1] * 100


        if satisfied_prob >= retention_threshold:
            st.markdown(f"""
                <div class="result-card success-card">
                    🎉 PASSENGER RETENTION<br>
                    <span style="font-size: 15px; font-weight: normal;">Satisfaction Probability: {satisfied_prob:.1f}% | Status: CHURN PROBABILITY NEUTRAL</span>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
                <div class="result-card error-card">
                    ⚠️ PASSENGER CHURN RISK<br>
                    <span style="font-size: 15px; font-weight: normal;">Dissatisfaction Probability: {100 - satisfied_prob:.1f}% | Risk: CHURN PROBABILITY POSSIBLE</span>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("### ⚙️ Experience Risk Mitigation & Strategic Insights")
            frontline_recs = []
            ceo_recs = []
            # 1. Operational & Strategic Rules based on scores
            if entertainment <= 3:
                frontline_recs.append("• 📺 <b>Inflight Entertainment Friction:</b> Offer a premium streaming pass or complimentary high-fidelity headphones for the next flight.")
                ceo_recs.append("• 📺 <b>Content & Hardware Upgrade Required:</b> Content library is outdated or screen hardware is failing. Recommend auditing IFE (In-Flight Entertainment) vendor contracts and expanding modern movie/game catalogs.")

            if seat_comfort <= 3:
                frontline_recs.append("• 🛋️ <b>Seat Comfort Issue:</b> Provide a complimentary travel pillow/blanket set immediately and flag the seat for physical inspection.")
                ceo_recs.append("• 🛋️ <b>Cabin Ergonomics Deficit:</b> High dissatisfaction with seating. Recommend accelerating the cabin retrofitting schedule and upgrading cushion materials in upcoming fleet maintenance cycles.")

            if food_drink <= 3:
                frontline_recs.append("• 🍱 <b>Catering Dissatisfaction:</b> Issue a $25 airport dining voucher or a complimentary premium onboard snack/beverage box.")
                ceo_recs.append("• 🍱 <b>Catering Quality Crisis:</b> Food/beverage scores are dropping. Recommend a comprehensive menu rotation and strict quality-assurance audits with the regional catering providers.")

            if cleanliness <= 3:
                frontline_recs.append("• 🧼 <b>Cabin Cleanliness Deficit:</b> Apologize proactively and assist the passenger with sanitizing wipes or a quick seat-pocket refresh if possible.")
                ceo_recs.append("• 🧼 <b>Turnaround Cleaning SLA Failure:</b> Cabin cleaning quality is substandard. Recommend reviewing ground handling SLAs (Service Level Agreements) and increasing cleaning window times between flights.")

            if leg_room_service <= 3:
                frontline_recs.append("• 🦿 <b>Leg Room Discomfort:</b> If available, execute an immediate complimentary upgrade to Extra Legroom or Comfort+ seating.")
                ceo_recs.append("• 🦿 <b>Fleet Configuration Constraint:</b> Pitch/legroom layout is causing churn. Recommend reviewing LOPA (Layout of Passenger Accommodations) for future aircraft orders to balance density and comfort.")

            if online_booking_service <= 3 or online_boarding <= 3:
                frontline_recs.append("• 📱 <b>Digital Interface Friction:</b> Manually process the customer's next check-in/booking and waive any offline processing fees.")
                ceo_recs.append("• 📱 <b>UX/UI Technical Debt:</b> Digital touchpoints are frustrating users. Recommend allocating budget to the IT/Product team for mobile app optimization and seamless digital boarding passes.")

            if wifi_service <= 3:
                frontline_recs.append("• 🌐 <b>Inflight Wi-Fi Failure:</b> Issue a full refund for the Wi-Fi package or credit 500 miles to the passenger's account.")
                ceo_recs.append("• 🌐 <b>Connectivity Infrastructure Deficit:</b> Inflight Wi-Fi reliability is low. Recommend upgrading satellite hardware (Ka/Ku-band) and renegotiating internet bandwidth contracts with provider networks.")

            if online_support <= 3:
                frontline_recs.append("• ☎️ <b>Substandard Online Support:</b> Escalate this passenger's profile to the Priority Support Tier for all future interactions.")
                ceo_recs.append("• ☎️ <b>Support Capacity & AI Training Deficit:</b> Digital support channels are bottlenecked. Recommend expanding the live-agent support team and retraining the automated chatbot conversational models.")

            if onboard_service <= 3:
                frontline_recs.append("• 🧑‍✈️ <b>Crew Service Failure:</b> Assign a dedicated senior crew member to personally oversee the passenger's remaining journey requirements.")
                ceo_recs.append("• 🧑‍✈️ <b>Crew Soft-Skills & Training Gap:</b> Service delivery is failing standards. Recommend launching mandatory crew hospitality retraining programs and internal performance audits.")

            if checkin_service <= 3 or baggage_handling <= 3:
                frontline_recs.append("• 🧳 <b>Airport Ground Friction:</b> Expedite baggage delivery tracking and offer priority lane tags for their next departure.")
                ceo_recs.append("• 🧳 <b>Ground Operations Hub Inefficiency:</b> Ground services are damaging satisfaction. Recommend tighter auditing of airport ground handling contractors or rewriting baggage KPIs.")

            if gate <= 3 or dep_val_time_convenient <= 3:
                frontline_recs.append("• 📅 <b>Schedule/Gate Inconvenience:</b> Provide complimentary lounge access to ease the airport transit stress.")
                ceo_recs.append("• 📅 <b>Network Planning & Slot Optimization:</b> Inconvenient gates or flight times are hurting retention. Recommend the Network Planning team optimize airport slot allocations and gate proximity agreements.")

            if arrival_delay_minutes > 15 or departure_delay_minutes > 15:
                max_delay = max(arrival_delay_minutes, departure_delay_minutes)
                frontline_recs.append(f"• ⏱️ <b>Operational Delay ({max_delay} min):</b> Issue an official apology letter along with airline miles proportional to the delay duration.")
                ceo_recs.append(f"• ⏱️ <b>OTP (On-Time Performance) Degradation:</b> Operational delays are triggering churn. Recommend a cross-functional review of maintenance dispatch reliability and ATC coordination.")

            if not frontline_recs:
                frontline_recs.append("• 🔄 <b>Composite Experience Friction:</b> Issue a standard 1,000 miles goodwill compensation and escalate account to the prioritized customer retention tier.")
                ceo_recs.append("• 🔄 <b>Micro-Friction Accumulation:</b> No single metric failed, but the combined minor friction points created an unsafe retention score. Recommend a holistic review of this target customer segment's journey map.")

            tab_frontline, tab_ceo = st.tabs(["👥 Frontline Actions (Support & Crew)", "👔 Executive Insights (CEO & Board)"])

            with tab_frontline:
                st.markdown(
                    f"<div class='action-box' style='background-color: #fffaf5; border-color: #ff9933; color: #2c3e50; text-align: left;'>"
                    f"<b>IMMEDIATE PASSENGER RETENTION STEPS:</b><br><br>"
                    f"{'<br><br>'.join(frontline_recs)}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with tab_ceo:
                st.markdown(
                    f"<div class='action-box' style='background-color: #f5f7fa; border-color: #2c3e50; color: #2c3e50; text-align: left;'>"
                    f"<b>LONG-TERM STRATEGIC RECOMMENDATIONS FOR MANAGEMENT:</b><br><br>"
                    f"{'<br><br>'.join(ceo_recs)}"
                    f"</div>",
                    unsafe_allow_html=True
                )
