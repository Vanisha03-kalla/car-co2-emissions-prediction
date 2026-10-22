# CarbonIQ — Complete `app.py`
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CarbonIQ | CO2 Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #020914;
        --panel: #071B2D;
        --border: #16445A;
        --text: #F1FAFF;
        --muted: #A8C5D8;
        --green: #00E6B0;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 80% 0%,
                rgba(0, 230, 176, 0.12),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #020914,
                #06182A 55%,
                #020914
            );
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #F4FBFF !important;
    }

    p, label {
        color: #C3D9E7;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #031323,
            #010812
        );
        border-right: 1px solid #12364D;
    }

    [data-testid="stSidebar"] * {
        color: #D9EDF7;
    }

    .hero {
        position: relative;
        overflow: hidden;
        min-height: 320px;
        padding: 35px;
        margin-bottom: 25px;
        border-radius: 22px;
        border: 1px solid rgba(0, 230, 176, 0.3);

        background:
            linear-gradient(
                90deg,
                rgba(1, 9, 20, 0.98) 0%,
                rgba(1, 12, 24, 0.88) 48%,
                rgba(1, 12, 24, 0.25) 100%
            ),
            url("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=1900&q=85")
            center 56% / cover;

        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.28);
    }

    .eyebrow {
        color: #00E6B0;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    .hero h1 {
        font-size: clamp(30px, 4vw, 49px);
        line-height: 1.15;
        letter-spacing: -1px;
        margin: 15px 0;
    }

    .hero h1 span {
        color: #00E6B0;
    }

    .hero-description {
        max-width: 550px;
        color: #D2E4EF;
        font-size: 14px;
        line-height: 1.8;
    }

    .pill {
        display: inline-block;
        padding: 8px 12px;
        margin: 12px 7px 0 0;
        border-radius: 25px;
        border: 1px solid rgba(0, 230, 176, 0.5);
        background: rgba(0, 230, 176, 0.08);
        color: #9DFFE7;
        font-size: 11px;
    }

    .metric-card {
        background: linear-gradient(
            145deg,
            #0A263B,
            #041321
        );
        border: 1px solid #16445A;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
    }

    .metric-label {
        color: #A8C5D8;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #00E6B0;
        font-family: 'Manrope', sans-serif;
        font-size: 27px;
        font-weight: 800;
    }

    .section-card {
        background: rgba(5, 27, 47, 0.8);
        border: 1px solid #123D55;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 16px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #00E6B0;
        color: #02121C;
        border: none;
        border-radius: 10px;
        font-weight: 800;
        min-height: 44px;
        transition: 0.2s ease;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #75FFD9;
        color: #02121C;
        border: none;
    }

    div[data-baseweb="input"] input {
        background-color: #071B2D;
        color: #F1FAFF;
    }

    div[data-baseweb="select"] > div {
        background-color: #071B2D;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #16445A;
        border-radius: 12px;
        overflow: hidden;
    }

    .footer {
        text-align: center;
        color: #7898AD;
        font-size: 12px;
        padding: 25px 0 5px;
        border-top: 1px solid #12364D;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DATASET AND MODEL
# --------------------------------------------------

DATA_PATH = Path(__file__).parent / "carbon_canada.csv"

FEATURES = [
    "Engine Size(L)",
    "Cylinders",
    "Fuel Consumption City (L/100 km)",
]

TARGET = "CO2 Emissions(g/km)"


@st.cache_data
def load_data(file_path, modified_time):
    data = pd.read_csv(file_path)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing these columns: "
            + ", ".join(missing_columns)
        )

    data = data[required_columns].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(subset=required_columns)

    return data


@st.cache_resource
def train_model(data):
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5

    return model, r2, mse, rmse, X_test, y_test, predictions


# --------------------------------------------------
# LOAD APPLICATION DATA
# --------------------------------------------------

st.sidebar.markdown("## 🌿 CarbonIQ")
st.sidebar.caption("Automotive Emissions Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Predict Emissions",
        "Dataset Explorer",
        "Model Performance",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning • Sustainability")


if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Please place carbon_canada.csv "
        "in the same folder as app.py."
    )
    st.code(
        "mini project- predicting co2 emissions from cars/\n"
        "├── app.py\n"
        "└── carbon_canada.csv"
    )
    st.stop()


try:
    data = load_data(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime,
    )

    if data.empty:
        st.error("The dataset has no valid rows after cleaning.")
        st.stop()

    (
        model,
        r2,
        mse,
        rmse,
        X_test,
        y_test,
        predictions,
    ) = train_model(data)

except Exception as error:
    st.error(f"Unable to load the dataset or train the model: {error}")
    st.stop()


# --------------------------------------------------
# REUSABLE HERO SECTION
# --------------------------------------------------




def show_hero():
    image_path = Path(__file__).parent / "carboniq_hero.png.png"

    if image_path.exists():
        st.image(
            str(image_path),
            use_container_width=True
        )
    else:
        st.error("Hero image not found. Check the image filename.")





def show_metric(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    show_hero()

    st.markdown("## Your emissions dashboard")
    st.write(
        "Explore the dataset, understand model performance, "
        "or estimate CO₂ emissions for a vehicle."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metric("DATA RECORDS", f"{len(data):,}")

    with col2:
        show_metric("INPUT FEATURES", str(len(FEATURES)))

    with col3:
        show_metric("MODEL R²", f"{r2:.3f}")

    with col4:
        show_metric("MODEL RMSE", f"{rmse:.2f}")

    st.markdown("## How CarbonIQ works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="section-card">
                <h3>01 · Enter details</h3>
                <p>Provide engine size, cylinder count and
                city fuel consumption.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">
                <h3>02 · ML prediction</h3>
                <p>The trained Linear Regression model estimates
                the vehicle's CO₂ emissions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="section-card">
                <h3>03 · Explore results</h3>
                <p>Review predictions, inspect the dataset and
                evaluate model performance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "This application provides estimates from a machine learning "
        "model. Predictions are not official vehicle certification values."
    )


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif page == "Predict Emissions":

    show_hero()

    st.markdown("## Vehicle emissions predictor")

    st.write(
        "Enter your vehicle specifications to estimate CO₂ emissions."
    )

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            engine_size = st.number_input(
                "Engine Size (L)",
                min_value=0.1,
                max_value=20.0,
                value=2.0,
                step=0.1,
                help="Engine displacement in litres.",
            )

        with col2:
            cylinders = st.selectbox(
                "Number of Cylinders",
                options=[3, 4, 5, 6, 8, 10, 12, 16],
                index=1,
            )

        with col3:
            fuel_consumption = st.number_input(
                "City Fuel Consumption (L/100 km)",
                min_value=0.1,
                max_value=100.0,
                value=10.0,
                step=0.1,
                help="Fuel consumed over 100 km of city driving.",
            )

        submitted = st.form_submit_button(
            "Calculate CO₂ Emissions",
            use_container_width=True,
        )

    if submitted:

        input_data = pd.DataFrame(
            [
                {
                    "Engine Size(L)": engine_size,
                    "Cylinders": cylinders,
                    "Fuel Consumption City (L/100 km)": fuel_consumption,
                }
            ]
        )

        predicted_emissions = float(model.predict(input_data)[0])

        st.session_state["prediction"] = predicted_emissions
        st.session_state["prediction_inputs"] = input_data

    if "prediction" in st.session_state:

        result = st.session_state["prediction"]

        st.markdown("---")
        st.markdown("### Your estimated result")

        result_col1, result_col2 = st.columns([2, 1])

        with result_col1:
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="metric-label">
                        ESTIMATED CARBON DIOXIDE EMISSIONS
                    </div>
                    <div class="metric-value"
                         style="font-size:42px;">
                        {result:,.1f}
                    </div>
                    <p>grams of CO₂ per kilometre (g/km)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_col2:
            st.markdown(
                """
                <div class="section-card">
                    <h3>What does this mean?</h3>
                    <p>This is the model's estimated CO₂ output
                    for the specifications you entered.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "The estimate depends on the training dataset and "
            "the selected model. It is not an official measurement."
        )

        if result < 0:
            st.warning(
                "The model produced a negative estimate. "
                "This is a limitation of Linear Regression and should "
                "not be interpreted as a physically valid emission value."
            )


# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------

elif page == "Dataset Explorer":

    st.markdown("## Dataset Explorer")
    st.write(
        "Explore the cleaned dataset used to train the prediction model."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        show_metric("VALID RECORDS", f"{len(data):,}")

    with col2:
        show_metric("FEATURES", str(len(FEATURES)))

    with col3:
        show_metric("MISSING VALUES", str(int(data.isna().sum().sum())))

    st.markdown("### Data preview")

    preview_rows = st.slider(
        "Number of rows to display",
        min_value=5,
        max_value=min(100, len(data)),
        value=min(10, len(data)),
    )

    st.dataframe(
        data.head(preview_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Descriptive statistics")

    st.dataframe(
        data.describe().round(2),
        use_container_width=True,
    )

    csv_data = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download cleaned dataset (CSV)",
        data=csv_data,
        file_name="carboniq_cleaned_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("### Feature relationships")

    selected_feature = st.selectbox(
        "Choose a feature",
        FEATURES,
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.scatter(
        data[selected_feature],
        data[TARGET],
        alpha=0.45,
    )

    ax.set_xlabel(selected_feature)
    ax.set_ylabel(TARGET)
    ax.set_title(f"{selected_feature} vs CO₂ Emissions")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.markdown("## Model Performance")
    st.write(
        "Evaluation results on the held-out test dataset."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        show_metric("R² SCORE", f"{r2:.4f}")

    with col2:
        show_metric("MEAN SQUARED ERROR", f"{mse:.2f}")

    with col3:
        show_metric("ROOT MEAN SQUARED ERROR", f"{rmse:.2f}")

    st.markdown("### Actual vs predicted emissions")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        y_test,
        predictions,
        alpha=0.55,
    )

    minimum = min(float(y_test.min()), float(predictions.min()))
    maximum = max(float(y_test.max()), float(predictions.max()))

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    ax.set_xlabel("Actual CO₂ Emissions (g/km)")
    ax.set_ylabel("Predicted CO₂ Emissions (g/km)")
    ax.set_title("Actual vs Predicted Values")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Model information")

    st.write("**Algorithm:** Linear Regression")
    st.write("**Preprocessing:** StandardScaler")
    st.write("**Test size:** 20%")
    st.write("**Random state:** 42")

    st.write("**Input features:**")

    for feature in FEATURES:
        st.write(f"- {feature}")

    st.write(f"**Target variable:** {TARGET}")

    st.info(
        "R² measures how well the model explains variation in the target. "
        "RMSE represents the typical prediction error scale in g/km. "
        "Evaluate these metrics alongside the dataset and test split."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------



# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div style="
        text-align: center;
        padding: 20px;
        color: #A8C5D8;
        font-size: 13px;
        border-top: 1px solid #16445A;
        margin-top: 30px;
    ">
    CarbonIQ 🌿<br>
Machine Learning for Sustainable Mobility<br>
        Built with ❤️ by Vanisha Kalla<br>
        ✦ Explore my GitHub → https://github.com/Vanisha03-kalla <br>

© 2026 CarbonIQ · Made with Python & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)



