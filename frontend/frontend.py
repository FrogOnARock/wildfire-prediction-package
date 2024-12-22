import streamlit as st
import requests

# Define the FastAPI base URL
BASE_URL = "http://127.0.0.1:8060"  # Update with your FastAPI URL if hosted remotely

st.title("Wildfire Prediction App")
st.write("Interact with the Wildfire Prediction API via this user-friendly interface.")

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Predict Cluster", "Predict Hectares Burned", "Train Models", "Shutdown"])

# Predict Cluster Tab
with tab1:
    st.header("Predict Wildfire Cluster")
    national_park = st.selectbox("Is it a National Park?", ["Y", "N"])
    temp_avg = st.number_input("Average Temperature (°C)", min_value=-40.0, max_value=40.0)
    temp_min = st.number_input("Minimum Temperature (°C)", min_value=-40.0, max_value=40.0)
    temp_max = st.number_input("Maximum Temperature (°C)", min_value=-40.0, max_value=40.0)
    precip = st.number_input("Precipitation (mm)", min_value=0.0, max_value=50.0)
    windspeed = st.number_input("Windspeed (km/h)", min_value=0.0, max_value=80.0)
    max_wind_gust = st.number_input("Max Wind Gust (km/h)", min_value=0.0, max_value=400.0)
    pressure = st.number_input("Pressure (hPa)", min_value=980.0, max_value=1100.0)
    cause = st.selectbox("Cause of Fire", ["H", "N", "H-PB", "U"])
    region = st.number_input("Region Identifier (1-13)", min_value=1, max_value=13, step=1)
    response = st.selectbox("Response Strategy", ["FUL", "MOD", "MON"])
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, step=1)

    if st.button("Predict Cluster"):
        # Make a POST request to the FastAPI predict/cluster endpoint
        payload = {
            "national_park": national_park,
            "temp_avg": temp_avg,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "precip": precip,
            "windspeed": windspeed,
            "max_wind_gust": max_wind_gust,
            "pressure": pressure,
            "cause": cause,
            "region": region,
            "response": response,
            "month": month
        }
        try:
            response = requests.post(f"{BASE_URL}/predict/cluster", json=payload)
            result = response.json()
            if response.status_code == 200:
                st.success(f"Predicted Cluster: {result['prediction']}")
            else:
                st.error(f"Error: {result['detail']}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# Predict Hectares Tab
with tab2:
    st.header("Predict Hectares Burned")
    cluster = st.number_input("Cluster Identifier (0-2)", min_value=0, max_value=2, step=1)
    # Reuse inputs for temperature, precip, etc., or create new fields here
    if st.button("Predict Hectares"):
        # Construct payload
        payload = {
            "cluster": cluster,
            "national_park": national_park,
            "temp_avg": temp_avg,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "precip": precip,
            "windspeed": windspeed,
            "max_wind_gust": max_wind_gust,
            "pressure": pressure,
            "cause": cause,
            "region": region,
            "response": response,
            "month": month,
        }
        try:
            response = requests.post(f"{BASE_URL}/predict/hectares", json=payload)
            result = response.json()
            if response.status_code == 200:
                st.success(f"Predicted Hectares Burned: {result['prediction']}")
            else:
                st.error(f"Error: {result['detail']}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# Train Models Tab
with tab3:
    st.header("Train Models")
    if st.button("Preprocess Data"):
        try:
            response = requests.post(f"{BASE_URL}/preprocess")
            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

    if st.button("Train Models"):
        try:
            response = requests.post(f"{BASE_URL}/train")
            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# Shutdown Tab
with tab4:
    st.header("Shutdown Application")
    if st.button("Shutdown"):
        try:
            response = requests.post(f"{BASE_URL}/shutdown")
            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Connection Error: {e}")