import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Car CO₂ Emissions Predictor",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F4F7F6;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #163A3D;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        color: #647575;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result-box {
        background-color: #E0F2E9;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #B6DCCB;
    }

    .result-label {
        color: #285B48;
        font-size: 18px;
    }

    .result-value {
        color: #163A3D;
        font-size: 42px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("carbon_canada.csv")

# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model():
    data = load_data()

    features = [
        "Engine Size(L)",
        "Cylinders",
        "Fuel Consumption City (L/100 km)"
    ]

    target = "CO2 Emissions(g/km)"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    return model, scaler, r2, mse, data

# -----------------------------
# LOAD APP RESOURCES
# -----------------------------
try:
    model, scaler, r2, mse, data = train_model()

except Exception as error:
    st.error(f"Unable to load the dataset or train the model: {error}")

    st.info(
        "Please check that carbon_canada.csv is in the same folder "
        "as app.py and that its column names match the code."
    )

    st.stop()

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<div class="main-title">🚗 Car CO₂ Emissions Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate vehicle carbon emissions using Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🌿 About This Project")

st.sidebar.write(
    "This application uses a Linear Regression model to estimate "
    "vehicle CO₂ emissions from engine size, cylinder count, "
    "and city fuel consumption."
)

st.sidebar.caption(
    "Predictions are estimates based on the training dataset."
)

# -----------------------------
# INPUT FORM
# -----------------------------
st.subheader("Enter Your Car Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    engine_size = st.number_input(
        "Engine Size (L)",
        min_value=0.1,
        max_value=20.0,
        value=2.0,
        step=0.1
    )

with col2:
    cylinders = st.number_input(
        "Number of Cylinders",
        min_value=1,
        max_value=16,
        value=4,
        step=1
    )

with col3:
    fuel_consumption = st.number_input(
        "City Fuel Consumption (L/100 km)",
        min_value=0.1,
        max_value=100.0,
        value=10.0,
        step=0.1
    )

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🌱 Predict CO₂ Emissions", use_container_width=True):

    input_data = pd.DataFrame(
        [[engine_size, cylinders, fuel_consumption]],
        columns=[
            "Engine Size(L)",
            "Cylinders",
            "Fuel Consumption City (L/100 km)"
        ]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.markdown("### Your Prediction")

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Estimated CO₂ Emissions</div>
            <div class="result-value">{prediction:.2f} g/km</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "This is a model estimate, not a certified measurement "
        "of the vehicle's emissions."
    )

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.divider()

st.subheader("📊 Model Performance")

metric1, metric2 = st.columns(2)

with metric1:
    st.metric("R² Score", f"{r2:.4f}")

with metric2:
    st.metric("Mean Squared Error", f"{mse:.2f}")

# -----------------------------
# DATA PREVIEW
# -----------------------------
with st.expander("🔍 Explore the Dataset"):
    st.write("Preview of the dataset used by this application:")
    st.dataframe(data.head(10), use_container_width=True)

    st.write(f"Total records: {len(data):,}")