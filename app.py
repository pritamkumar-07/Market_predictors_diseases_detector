import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.market_predictor import (
    load_market_data,
    build_market_features,
    predict_market_price,
    market_summary,
)
from src.disease_detector import load_disease_model, predict_disease

ROOT = Path(__file__).resolve().parent
MARKET_MODEL = ROOT / "models" / "market_model.joblib"
DISEASE_MODEL = ROOT / "models" / "disease_cnn.keras"
CLASS_FILE = ROOT / "models" / "class_names.json"

st.set_page_config(
    page_title="Farmer AI Assistant",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Market Predictor & Disease Detector for Farmer")
st.caption("Python • Pandas • NumPy • Statistics • ML • Deep Learning • CNN")

market_df = load_market_data(ROOT / "data" / "market_data.csv")

tab1, tab2, tab3 = st.tabs(["📈 Market Predictor", "🍃 Disease Detector", "📊 Dataset Analysis"])

with tab1:
    st.header("Market Price Prediction")

    if not MARKET_MODEL.exists():
        st.warning("Market model not found. Run: python scripts/train_market_model.py")
    else:
        model = joblib.load(MARKET_MODEL)

        col1, col2, col3 = st.columns(3)
        crops = sorted(market_df["Crop"].dropna().unique().tolist())
        markets = sorted(market_df["Market"].dropna().unique().tolist())
        seasons = sorted(market_df["Season"].dropna().unique().tolist())

        with col1:
            crop = st.selectbox("Crop", crops)
        with col2:
            market = st.selectbox("Market", markets)
        with col3:
            season = st.selectbox("Season", seasons)

        crop_rows = market_df[market_df["Crop"] == crop].sort_values("Date")
        default_prev = float(crop_rows["Price"].iloc[-1]) if not crop_rows.empty else 2000.0
        previous_price = st.number_input(
            "Previous market price (₹/quintal)",
            min_value=0.0,
            value=default_prev,
            step=10.0,
        )

        quantity = st.number_input(
            "Quantity available (quintal)",
            min_value=0.0,
            value=100.0,
            step=10.0,
        )

        date = st.date_input("Prediction date")

        if st.button("Predict Market Price", type="primary"):
            input_df = pd.DataFrame([{
                "Crop": crop,
                "Date": pd.Timestamp(date),
                "Market": market,
                "Season": season,
                "PreviousPrice": previous_price,
                "Quantity": quantity,
            }])
            X = build_market_features(input_df)
            prediction = predict_market_price(model, X)

            st.success(f"Estimated market price: ₹{prediction:,.2f} per quintal")
            st.info("This is a model-based estimate, not a guaranteed future market price.")

with tab2:
    st.header("Crop Disease Detector")

    if not DISEASE_MODEL.exists() or not CLASS_FILE.exists():
        st.warning(
            "No trained disease CNN is available yet. Add a real labeled crop-leaf "
            "dataset and run: python scripts/train_disease_cnn.py"
        )
        st.markdown(
            """
            **Expected dataset structure**
            ```
            disease_dataset/
            ├── train/<class_name>/*.jpg
            ├── validation/<class_name>/*.jpg
            └── test/<class_name>/*.jpg
            ```
            """
        )
    else:
        uploaded = st.file_uploader(
            "Upload a crop-leaf image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded crop image", width=420)

            if st.button("Detect Disease", type="primary"):
                model = load_disease_model(DISEASE_MODEL)
                class_names = json.loads(CLASS_FILE.read_text())
                label, confidence = predict_disease(model, image, class_names)

                st.success(f"Predicted class: {label}")
                st.metric("Model confidence", f"{confidence * 100:.2f}%")
                st.caption(
                    "The prediction is meaningful only when the model has been "
                    "trained on a representative labeled crop-disease dataset."
                )

with tab3:
    st.header("Market Dataset & Statistical Analysis")

    st.dataframe(market_df, use_container_width=True)

    summary = market_summary(market_df)
    st.subheader("Descriptive Statistics")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Average Price by Crop")
    avg = market_df.groupby("Crop", as_index=False)["Price"].mean()
    st.bar_chart(avg.set_index("Crop"))

    st.subheader("Price Trend")
    trend = market_df.groupby("Date", as_index=False)["Price"].mean().set_index("Date")
    st.line_chart(trend)
