import streamlit as st
import pandas as pd
import joblib
st.markdown("""
<style>
.footer {
    text-align: center;
    margin-top: 50px;
    color: #888;
    font-size: 14px;
}
.footer strong {
    color: #ccc;
}
</style>
""", unsafe_allow_html=True)
# ─── PAGE CONFIG ─────────────────────────
st.set_page_config(
    page_title="Customer Segmentation",
    layout="centered"
)

# ─── TITLE ───────────────────────────────
st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment")

# ─── LOAD MODELS ─────────────────────────
kmeans = joblib.load("KMEANS_MODEL.PKL")
scaler = joblib.load("SCALER.PKL")
segment_map = joblib.load("segment_map.pkl")

# ─── INPUTS ──────────────────────────────
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income(in dollars /Annual)", min_value=0, max_value=200000, value=5000)
total_spending = st.number_input("Total Spending", min_value=0, max_value=50000, value=1000)

num_web_purchases = st.number_input("Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Web Visits per Month", min_value=0, max_value=100, value=10)

recency = st.number_input("Days Since Last Purchase", min_value=0, max_value=365, value=30)

# ─── DATAFRAME ───────────────────────────
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "Recency": [recency],
    "NumWebVisitsMonth": [num_web_visits]
})

# ─── SCALE ───────────────────────────────
input_scaled = scaler.transform(input_data)

# ─── RECOMMENDATION FUNCTION ─────────────
def get_recommendation(segment):
    if "High Value" in segment:
        return "Focus on loyalty programs and premium offers."
    elif "Low Value" in segment:
        return "Provide discounts and budget-friendly options."
    elif "At Risk" in segment:
        return "Re-engage with campaigns and offers."
    elif "Digital" in segment:
        return "Increase digital marketing and online ads."
    else:
        return "Apply general marketing strategies."

# ─── PREDICT ─────────────────────────────
if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    segment = segment_map.get(cluster, "Unknown")

    st.success(f"Predicted Segment: {segment}")

    recommendation = get_recommendation(segment)
    st.info(f"Recommendation: {recommendation}")

# ─── FOOTER ──────────────────────────────
st.markdown("""
<div class="footer">
    ✦ made with love & late nights by ✦<br>
    <strong>Asrar Farooq Wani</strong><br>
     with <strong>coffee and Pringles</strong>
</div>
""", unsafe_allow_html=True)
