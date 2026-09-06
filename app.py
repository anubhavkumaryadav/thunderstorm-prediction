import urllib.parse
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st

st.set_page_config(
    page_title="Thunderstorm Predictor", page_icon="⛈️", layout="centered"
)


# Safety guidelines block
def show_safety_guidelines():
    st.error("### ⚠️ Preventive Measures & Guidelines")
    st.markdown("""
    - **Seek Sturdy Shelter:** Immediately go indoors or inside a hard-topped vehicle.
    - **Avoid Isolated Structures:** Stay far away from open fields, high trees, telephone poles, and metal fences.
    - **Stay Away From Water:** Exit swimming pools, lakes, and open water bodies immediately.
    - **Indoor Precautions:** Unplug delicate electronics (laptops, TVs) and avoid running tap water during active lightning strikes.
    - **30/30 Rule:** If the time between lightning and thunder is under 30 seconds, stay indoors until 30 minutes after the last thunderclap.
    """)


# Train model
@st.cache_resource
def train_model():
    data = pd.read_csv("data.csv")
    X = data[["temperature", "humidity", "pressure", "wind_speed"]]
    y = data["thunderstorm"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


model = train_model()

st.title("⛈️ Thunderstorm Detection & Weather Hub")
st.write(
    "Check thunderstorm chances manually or fetch real-time live weather by city."
)

tab1, tab2 = st.tabs(
    ["1️⃣ Live City Weather & Link", "2️⃣ Manual Input Prediction"]
)

with tab1:
    st.subheader("Option 1: Live City Weather Detection")

    city = st.text_input(
        "Enter City Name:", value="Mumbai", placeholder="e.g. Mumbai, Delhi, Tokyo"
    )

    if st.button("Fetch Live Weather & Detect", key="btn_live"):
        clean_city = city.strip()
        if not clean_city:
            st.error("Please type a city name.")
        else:
            with st.spinner(f"Getting live weather data for {clean_city}..."):
                # Geocoding API: find coordinates
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(clean_city)}&count=1&language=en&format=json"
                geo_resp = requests.get(geo_url).json()

                if "results" in geo_resp and len(geo_resp["results"]) > 0:
                    location = geo_resp["results"][0]
                    lat = location["latitude"]
                    lon = location["longitude"]
                    city_name = location["name"]
                    country = location.get("country", "")

                    # Weather API: fetch live conditions
                    weather_url = (
                        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                        f"&current=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
                    )
                    w_resp = requests.get(weather_url).json()
                    curr = w_resp["current"]

                    live_temp = curr["temperature_2m"]
                    live_humidity = curr["relative_humidity_2m"]
                    live_pressure = curr["surface_pressure"]
                    live_wind = curr["wind_speed_10m"]

                    st.success(
                        f"Showing real-time weather for **{city_name}, {country}**"
                    )

                    # 1. LIVE DATA DISPLAY CARDS
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Temperature", f"{live_temp} °C")
                    col_b.metric("Humidity", f"{live_humidity}%")
                    col_c.metric("Pressure", f"{live_pressure} hPa")
                    col_d.metric("Wind Speed", f"{live_wind} km/h")

                    # 2. CLICKABLE EXTERNAL LINKS
                    st.write("---")
                    st.write(
                        "🔗 **Direct Links to view official live weather:**"
                    )

                    google_weather_url = f"https://www.google.com/search?q={urllib.parse.quote(city_name)}+weather"
                    accuweather_search = f"https://www.accuweather.com/en/search-locations?query={urllib.parse.quote(city_name)}"

                    link_col1, link_col2 = st.columns(2)
                    with link_col1:
                        st.link_button(
                            f"🌐 View {city_name} on Google Weather",
                            google_weather_url,
                        )
                    with link_col2:
                        st.link_button(
                            f"📡 View {city_name} on AccuWeather",
                            accuweather_search,
                        )

                    # 3. RUN THE MODEL
                    features = [
                        [live_temp, live_humidity, live_pressure, live_wind]
                    ]
                    prob = model.predict_proba(features)[0][1] * 100
                    prediction = model.predict(features)[0]

                    st.write("---")
                    st.subheader("Model Thunderstorm Risk Assessment")
                    st.metric(
                        label="Calculated Thunderstorm Probability",
                        value=f"{prob:.1f}%",
                    )
                    st.progress(int(prob))

                    if prob >= 50 or prediction == 1:
                        st.warning(
                            "⚡ High thunderstorm risk! Review precautions below:"
                        )
                        show_safety_guidelines()
                    else:
                        st.success(
                            "☀️ Low thunderstorm risk. Conditions appear clear!"
                        )
                else:
                    st.error(
                        f"Could not find '{clean_city}'. Check the spelling and try again."
                    )

with tab2:
    st.subheader("Option 2: Enter Weather Values Manually")

    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input(
            "Temperature (°C)", value=30.0, step=0.5, key="m_temp"
        )
        pressure = st.number_input(
            "Pressure (hPa)", value=1005.0, step=1.0, key="m_press"
        )
    with col2:
        humidity = st.number_input(
            "Humidity (%)",
            value=75.0,
            step=1.0,
            min_value=0.0,
            max_value=100.0,
            key="m_hum",
        )
        wind_speed = st.number_input(
            "Wind Speed (km/h)", value=15.0, step=0.5, key="m_wind"
        )

    if st.button("Check Chances (Manual)", key="btn_manual"):
        features = [[temp, humidity, pressure, wind_speed]]
        prob = model.predict_proba(features)[0][1] * 100
        prediction = model.predict(features)[0]

        st.divider()
        st.metric(
            label="Calculated Thunderstorm Probability", value=f"{prob:.1f}%"
        )
        st.progress(int(prob))

        if prob >= 50 or prediction == 1:
            st.warning("⚡ High thunderstorm risk! Review precautions below:")
            show_safety_guidelines()
        else:
            st.success("☀️ Low thunderstorm risk. Conditions appear clear!")
