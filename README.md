# ✈️ SkyInsight: Predictive Passenger Experience Analytics & Loyalty Modeling

## 🚀 Live Application  
🌐 Try the App  
👉 https://skyinsight-predictive-passenger-experience-analytics-loyalty-m.streamlit.app/


## 📊 Project Presentation
Executive presentation covering the business problem, analytical methodology, machine learning results, key insights, and customer retention recommendations.
👉 [Download PDF](predictive_analytics_pres.pdf)

---

### 🎯 Overview
An end-to-end Machine Learning system designed to predict airline passenger satisfaction, detect early-warning churn signals, and optimize customer retention strategies.

The project transforms traditional reactive survey analytics into a proactive, decision-oriented retention engine used for real-time customer experience insights.

### 📊 Performance
- **Accuracy:** 96.1%
- **ROC-AUC:** 99.5%
- **Training Data:** 130,000+ passenger records
- **Model Type**: Supervised Classification

---

## 🎯 Executive Summary & Business Challenge
* **The Core Crisis:** While 82% of the airline's passenger base consists of high-value, loyal customers, nearly **31% within this premium segment are dissatisfied**, presenting a massive hidden churn threat.
* **The Financial ROI:** Retaining an existing passenger is **5x to 7x more cost-effective** than acquiring a new one. By preventing loyalty degradation, this system directly safeguards recurring airline revenue and optimizes capital expenditure (CapEx).
* **Strategic Shift:** Moving the organization from *Reactive feedback analysis* (e.g., traditional retrospective NPS) to *Proactive real-time customer interventions*.

---

## 📊 Strategic Business Insights & Model Outcomes

By evaluating complex, non-linear feature dependencies, the predictive model overcame data multicollinearity blind spots and established a strategic investment roadmap:

1. **In-Flight Comfort (Priority #1 | 54% Impact):** Anchored by *Inflight Entertainment* and *Seat Comfort*. For our core professional audience (aged 38–60 flying Business Class), physical comfort is a baseline expectation, not a premium bonus. If the screens fail, loyalty is completely compromised.
2. **Digital Experience (Priority #2 | 25% Impact):** Encompasses *Online Booking*, *Online Boarding*, *Online Support*, and *Wi-Fi Service*. 
3. **Airport & Crew Service (Priority #3 | 13% Impact):** Encompasses *Onboard Service*, *Check-in Service*, and *Baggage Handling*.
4. **Flight Reliability (Priority #4 | 8% Impact):** Driven by *Gate Location* and *Departure/Arrival Time Convenience*.

### ⚠️ Critical Friction Thresholds Identified:
* **The 4-Star Rule:** Customer satisfaction does not scale linearly. Passengers perceive a 3-star rating almost as negatively as a 1-star rating. True customer retention is triggered **only when service delivery consistently hits 4 or 5 stars**.
* **The Point of No Return:** Flight delays exceeding **15 minutes** immediately flip the passenger majority to dissatisfied. Once a delay surpasses **120 minutes (2 hours)**, dissatisfaction spikes to 63% and permanently plateaus. 

---

## ⚙️ Model Performance & Technical Architecture

The production pipeline compares three diagnostic baselines trained on over **130,000 historical passenger surveys**:



| Model | Overall Accuracy | Prediction Precision | Risk Detection (Recall) | F1-Score | Model Reliability (ROC-AUC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | 96.1% | 97.1% | 95.7% | 96.4% | 99.5% |
| **Random Forest** | 96.0% | 96.9% | 95.6% | 96.3% | 99.4% |
| **Logistic Regression** | 83.5% | 84.6% | 85.0% | 84.8% | 90.9% |

### Machine Learning Insights:
* **Precision (97.1%):** High precision ensures that the business rarely fires false alarms, preventing the waste of retention budgets (such as miles or discounts) on already satisfied customers.
* **Recall (95.7%):** Captures almost the entire "Danger Zone" population of dissatisfied, loyal passengers before they switch to a competitor.

---

## 🚀 Interactive Deployment (Streamlit App)
To transition data insights into immediate corporate actions, a web-based decision application was built using **Streamlit** and secured via an **ngrok** tunnel. 

The application allows customer experience managers or gate agents to input passenger parameters during friction events (such as flight delays or broken IFE systems). The model outputs a real-time **Churn Risk Level** or **Retention Confidence Score**, enabling instant, automated mitigation offers.

---

## 🛠️ Repository Structure
* `airline_satisfaction_analysis.ipynb` — Comprehensive notebook detailing the data processing, Exploratory Data Analysis (EDA), feature engineering, and model validation pipeline.
* `app.py` — The interactive Streamlit application serving the core inference model.
* `airline_rf_pipeline.pkl` — Serialized machine learning model pipeline ready for deployment.
* `requirements.txt` — Environment specifications.
* `predictive_analytics_pres.pdf` — executive presentation covering the business problem, methodology, model performance, key insights, and retention strategy recommendations.

---
