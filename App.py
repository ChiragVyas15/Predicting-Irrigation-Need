from pathlib import Path

import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "final_model.cbm"
TRAIN_PATH = ROOT / "train.csv"
SAMPLE_PATH = ROOT / "sample_submission.csv"

RAW_FEATURES = [
    "Soil_Type",
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Field_Area_hectare",
    "Mulching_Used",
    "Previous_Irrigation_mm",
    "Region",
]

DROP_COLS = [
    "Soil_Moisture",
    "Rainfall_mm",
    "Previous_Irrigation_mm",
    "Temperature_C",
    "Sunlight_Hours",
    "Humidity",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Soil_pH",
]


@st.cache_resource
def load_model() -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_options() -> dict[str, list[str]]:
    df = pd.read_csv(TRAIN_PATH, usecols=RAW_FEATURES)
    categorical_cols = df.select_dtypes(include=["object"]).columns
    return {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in categorical_cols
    }


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Moisture_Deficit"] = 100 - df["Soil_Moisture"]
    df["Water_Availability"] = df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
    df["Evaporation"] = (
        df["Temperature_C"] * df["Sunlight_Hours"] / (df["Humidity"] + 1)
    )
    df["Heat_Stress"] = df["Temperature_C"] * df["Wind_Speed_kmh"]
    df["Water_per_Area"] = df["Previous_Irrigation_mm"] / df["Field_Area_hectare"]
    df["Moisture_Rain"] = df["Soil_Moisture"] * df["Rainfall_mm"]
    df["pH_Category"] = pd.cut(
        df["Soil_pH"],
        bins=[0, 6.5, 7.5, 14],
        labels=["Acidic", "Neutral", "Alkaline"],
    )
    return df.drop(columns=DROP_COLS)


def prepare_for_prediction(df: pd.DataFrame, model: CatBoostClassifier) -> pd.DataFrame:
    missing_cols = [col for col in RAW_FEATURES if col not in df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"Missing required columns: {missing}")

    processed = add_features(df[RAW_FEATURES])
    X = processed[model.feature_names_].copy()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_features:
        X[col] = X[col].astype(str)
    return X


def predict(df: pd.DataFrame, model: CatBoostClassifier) -> pd.Series:
    X = prepare_for_prediction(df, model)
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    predictions = model.predict(Pool(X, cat_features=cat_features)).ravel()
    return pd.Series(predictions, name="Irrigation_Need")


def build_single_input(options: dict[str, list[str]]) -> pd.DataFrame:
    with st.form("single_prediction_form"):
        st.subheader("Field Details")
        left, middle, right = st.columns(3)

        with left:
            soil_type = st.selectbox("Soil type", options["Soil_Type"])
            soil_ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, 0.01)
            soil_moisture = st.number_input("Soil moisture", 0.0, 100.0, 35.0, 0.01)
            organic_carbon = st.number_input("Organic carbon", 0.0, 10.0, 1.0, 0.01)
            electrical_conductivity = st.number_input(
                "Electrical conductivity", 0.0, 10.0, 2.5, 0.01
            )
            field_area = st.number_input("Field area hectare", 0.01, 100.0, 5.0, 0.01)
            region = st.selectbox("Region", options["Region"])

        with middle:
            temperature = st.number_input("Temperature C", -10.0, 60.0, 25.0, 0.01)
            humidity = st.number_input("Humidity", 0.0, 100.0, 60.0, 0.01)
            rainfall = st.number_input("Rainfall mm", 0.0, 3000.0, 700.0, 0.01)
            sunlight = st.number_input("Sunlight hours", 0.0, 24.0, 7.0, 0.01)
            wind_speed = st.number_input("Wind speed kmh", 0.0, 100.0, 10.0, 0.01)
            previous_irrigation = st.number_input(
                "Previous irrigation mm", 0.0, 500.0, 50.0, 0.01
            )

        with right:
            crop_type = st.selectbox("Crop type", options["Crop_Type"])
            crop_growth_stage = st.selectbox(
                "Crop growth stage", options["Crop_Growth_Stage"]
            )
            season = st.selectbox("Season", options["Season"])
            irrigation_type = st.selectbox("Irrigation type", options["Irrigation_Type"])
            water_source = st.selectbox("Water source", options["Water_Source"])
            mulching_used = st.selectbox("Mulching used", options["Mulching_Used"])

        submitted = st.form_submit_button("Predict")

    data = pd.DataFrame(
        [
            {
                "Soil_Type": soil_type,
                "Soil_pH": soil_ph,
                "Soil_Moisture": soil_moisture,
                "Organic_Carbon": organic_carbon,
                "Electrical_Conductivity": electrical_conductivity,
                "Temperature_C": temperature,
                "Humidity": humidity,
                "Rainfall_mm": rainfall,
                "Sunlight_Hours": sunlight,
                "Wind_Speed_kmh": wind_speed,
                "Crop_Type": crop_type,
                "Crop_Growth_Stage": crop_growth_stage,
                "Season": season,
                "Irrigation_Type": irrigation_type,
                "Water_Source": water_source,
                "Field_Area_hectare": field_area,
                "Mulching_Used": mulching_used,
                "Previous_Irrigation_mm": previous_irrigation,
                "Region": region,
            }
        ]
    )
    return data if submitted else pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title="Irrigation Need Predictor", layout="wide")

    st.title("Irrigation Need Predictor")
    st.caption("CatBoost model loaded from final_model.cbm")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}")
        st.stop()

    model = load_model()
    options = load_options()

    single_tab, batch_tab = st.tabs(["Single Prediction", "Batch CSV"])

    with single_tab:
        single_df = build_single_input(options)
        if not single_df.empty:
            result = predict(single_df, model).iloc[0]
            st.metric("Predicted irrigation need", result)

    with batch_tab:
        uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                output = pd.DataFrame()
                output["id"] = (
                    input_df["id"]
                    if "id" in input_df.columns
                    else range(len(input_df))
                )
                output["Irrigation_Need"] = predict(input_df, model)

                st.success(f"Predicted {len(output):,} rows")
                st.dataframe(output.head(20), use_container_width=True)
                st.download_button(
                    "Download submission.csv",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="submission.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(str(exc))

        if SAMPLE_PATH.exists():
            st.caption("Output format: id, Irrigation_Need")


if __name__ == "__main__":
    main()
