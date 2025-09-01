# app.py
import streamlit as st
import requests

st.set_page_config(page_title="Gift Recommender", layout="centered")
st.title("🎁 AI Gift Recommender")

API_URL = "http://127.0.0.1:8000/recommend"

occasion = st.text_input("Occasion", "Birthday")
age = st.number_input("Age", min_value=1, max_value=100, value=22)
gender = st.selectbox("Gender", ["any","male","female"])
interests = st.text_input("Interests (comma separated)")
budget_min = st.number_input("Budget Min (Rs.)", min_value=0, value=300, step=50)
budget_max = st.number_input("Budget Max (Rs.)", min_value=0, value=2000, step=50)

if st.button("Get Recommendations"):
    payload = {
        "occasion": occasion,
        "age": int(age),
        "gender": gender,
        "interests": interests,
        "budget_min": int(budget_min),
        "budget_max": int(budget_max)
    }
    with st.spinner("Generating recommendations..."):
        try:
            r = requests.post(API_URL, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if "recommendations" in data:
                    st.markdown("### ✨ Suggested Gifts")
                    st.code(data["recommendations"])
                else:
                    st.error(data.get("error", "No response"))
            else:
                st.error(f"API error: {r.status_code} - {r.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
