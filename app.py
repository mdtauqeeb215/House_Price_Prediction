
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EstateIQ | House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- PROFESSIONAL CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f4f7fb;
}

[data-testid="stHeader"] {
    background: rgba(244, 247, 251, 0.9);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.hero {
    background: linear-gradient(120deg, #101c3a 0%, #233d70 100%);
    padding: 34px;
    border-radius: 22px;
    color: white;
    margin-bottom: 26px;
    box-shadow: 0 12px 30px rgba(24, 44, 83, 0.12);
}

.hero-label {
    color: #9ebcff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
    margin: 12px 0 8px 0;
}

.hero p {
    color: #d6e0f4;
    font-size: 15px;
    margin-bottom: 0;
}

.section-title {
    font-size: 21px;
    font-weight: 750;
    color: #172442;
    margin: 12px 0 4px 0;
}

.section-subtitle {
    font-size: 13px;
    color: #6b7890;
    margin-bottom: 18px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 4px 15px rgba(20, 40, 80, 0.035);
}



[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p {
    color: #334155 !important;
    -webkit-text-fill-color: #334155 !important;
    opacity: 1 !important;
    visibility: visible !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}


div[data-testid="stMetricValue"] {
    color: #172442 !important;
    font-weight: 750 !important;
}

div[data-testid="stMetricValue"] {
    color: #172442;
    font-weight: 750;
}

.panel {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 18px;
}

.result-card {
    background: linear-gradient(135deg, #e8f5ee, #f5fbf7);
    border: 1px solid #cce9d8;
    padding: 24px;
    border-radius: 18px;
    margin-top: 18px;
}

.result-label {
    color: #387653;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-price {
    color: #145c38;
    font-size: 34px;
    font-weight: 800;
    margin: 8px 0;
}

.result-note {
    color: #547563;
    font-size: 13px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #244b91;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #19396f;
    color: white;
    border: none;
}

div[data-testid="stSidebar"] {
    background: #101c35;
}

div[data-testid="stSidebar"] * {
    color: #eef3ff;
}

.footer {
    text-align: center;
    color: #8390a5;
    font-size: 12px;
    padding-top: 24px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    data = pd.read_csv("house_data.csv")

    required = ["area", "bedrooms", "bathrooms", "location", "price"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            f"Missing columns in house_data.csv: {', '.join(missing)}"
        )

    data = data[required].copy()
    data["location"] = data["location"].astype("string").str.strip()

    for column in ["area", "bedrooms", "bathrooms", "price"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=required)
    data = data[data["location"] != ""]

    return data


@st.cache_resource
def train_model(data):
    features = ["area", "bedrooms", "bathrooms", "location"]
    X = data[features]
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("location", OneHotEncoder(handle_unknown="ignore"), ["location"])
        ],
        remainder="passthrough",
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regression", LinearRegression()),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions) if len(y_test) > 1 else float("nan"),
    }

    actual_vs_predicted = pd.DataFrame({
        "Actual Price": y_test.to_numpy(),
        "Predicted Price": predictions,
    })

    return model, metrics, actual_vs_predicted


try:
    df = load_data()

    if len(df) < 5:
        st.error("Model train karne ke liye dataset mein kam se kam 5 valid rows honi chahiye.")
        st.stop()

    model, metrics, comparison = train_model(df)

except Exception as error:
    st.error(f"Dataset ya model load nahi ho paya: {error}")
    st.stop()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🏡 EstateIQ")
    st.caption("HOUSE PRICE INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Price Prediction", "Dataset"],
        label_visibility="visible",
    )

    st.divider()
    st.caption("MODEL INFORMATION")
    st.write("Algorithm: Linear Regression")
    st.write(f"Dataset records: {len(df):,}")
    st.write(f"Locations: {df['location'].nunique()}")

    st.divider()
    st.caption("Built with Python & Streamlit")


# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="hero-label">SMART REAL ESTATE ANALYTICS</div>
    <h1>House Price Intelligence</h1>
    <p>Explore housing data, understand model performance, and estimate
    property prices with machine learning.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.markdown('<div class="section-title">Market Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A quick look at your housing dataset and model performance.</div>',
        unsafe_allow_html=True,
    )

    average_price = df["price"].mean()
    median_price = df["price"].median()
    average_area = df["area"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Average House Price", f"₹{average_price / 100000:.2f} L")
    c2.metric("Median House Price", f"₹{median_price / 100000:.2f} L")
    c3.metric("Mean Absolute Error", f"₹{metrics['mae'] / 100000:.2f} L")
    c4.metric("R² Score", f"{metrics['r2']:.3f}" if pd.notna(metrics["r2"]) else "N/A")

    st.markdown("---")

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown('<div class="section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">How the model performed on the test data.</div>',
            unsafe_allow_html=True,
        )

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(
            comparison["Actual Price"] / 100000,
            comparison["Predicted Price"] / 100000,
            alpha=0.75,
            edgecolors="white",
        )

        lower = min(
            comparison["Actual Price"].min(),
            comparison["Predicted Price"].min(),
        ) / 100000

        upper = max(
            comparison["Actual Price"].max(),
            comparison["Predicted Price"].max(),
        ) / 100000

        ax.plot([lower, upper], [lower, upper], linestyle="--")
        ax.set_xlabel("Actual Price (₹ Lakhs)")
        ax.set_ylabel("Predicted Price (₹ Lakhs)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown('<div class="section-title">Location Insights</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Average listed price by location.</div>',
            unsafe_allow_html=True,
        )

        location_prices = (
            df.groupby("location")["price"]
            .mean()
            .sort_values(ascending=True)
        )

        st.bar_chart(location_prices / 100000, horizontal=True, x_label="Average price (₹ Lakhs)")

        st.markdown("**Dataset snapshot**")
        st.write(f"- Total properties: {len(df):,}")
        st.write(f"- Average area: {average_area:,.0f} sq. ft.")
        st.write(f"- Number of locations: {df['location'].nunique()}")


# ---------------- PRICE PREDICTION ----------------
elif page == "Price Prediction":
    st.markdown('<div class="section-title">Estimate Your Property Value</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Enter the property details to generate a machine-learning estimate.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            area = st.number_input(
                "Built-up area (sq. ft.)",
                min_value=100,
                max_value=50000,
                value=1200,
                step=50,
            )

            bedrooms = st.selectbox(
                "Number of bedrooms",
                options=list(range(1, 11)),
                index=2,
            )

        with right:
            bathrooms = st.selectbox(
                "Number of bathrooms",
                options=list(range(1, 11)),
                index=1,
            )

            locations = sorted(df["location"].astype(str).unique().tolist())

            location = st.selectbox(
                "Property location",
                options=locations,
            )

        submitted = st.form_submit_button("✨ Predict House Price")

    if submitted:
        new_house = pd.DataFrame([{
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location": location,
        }])

        predicted_price = float(model.predict(new_house)[0])

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Property Value</div>
            <div class="result-price">₹{predicted_price / 100000:,.2f} Lakhs</div>
            <div class="result-note">
                Approx. ₹{predicted_price:,.0f} • {area:,} sq. ft. •
                {bedrooms} bedroom(s) • {location}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "This is a machine-learning estimate, not a formal property valuation. "
            "Actual prices may vary with property condition, locality, and market trends."
        )


# ---------------- DATASET ----------------
elif page == "Dataset":
    st.markdown('<div class="section-title">Housing Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Preview and inspect the data used to train the model.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Average Area", f"{df['area'].mean():,.0f} sq. ft.")
    c3.metric("Unique Locations", f"{df['location'].nunique()}")

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("**Summary statistics**")
    
st.dataframe(
    df.describe(include="all").astype(str),
    use_container_width=True
)



# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    ESTATEIQ • HOUSE PRICE PREDICTION • POWERED BY MACHINE LEARNING
</div>
""", unsafe_allow_html=True)
