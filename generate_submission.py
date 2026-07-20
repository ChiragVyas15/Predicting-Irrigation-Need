from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool


ROOT = Path(__file__).resolve().parent

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


def main() -> None:
    train_df = pd.read_csv(ROOT / "train.csv")
    test_df = pd.read_csv(ROOT / "test.csv")
    sample_submission = pd.read_csv(ROOT / "sample_submission.csv")

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    X = train_df.drop(columns=["Irrigation_Need", "id"])
    y = train_df["Irrigation_Need"]
    X_test = test_df.drop(columns=["id"])

    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_features:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=100,
    )
    model.fit(Pool(X, y, cat_features=cat_features))

    predictions = model.predict(Pool(X_test, cat_features=cat_features)).ravel()
    submission = sample_submission.copy()
    submission["Irrigation_Need"] = predictions
    submission.to_csv(ROOT / "submission.csv", index=False)

    print("Created submission.csv")
    print(submission.head())
    print(submission["Irrigation_Need"].value_counts())


if __name__ == "__main__":
    main()
