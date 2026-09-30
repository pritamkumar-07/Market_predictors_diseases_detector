from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.market_predictor import build_market_features, load_market_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "market_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = load_market_data(DATA)

crop_map = {v: i for i, v in enumerate(sorted(df["Crop"].unique()))}
market_map = {v: i for i, v in enumerate(sorted(df["Market"].unique()))}
season_map = {v: i for i, v in enumerate(sorted(df["Season"].unique()))}

X = build_market_features(df, crop_map, market_map, season_map)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(
    n_estimators=250,
    random_state=42,
    max_depth=10,
    min_samples_leaf=2,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

joblib.dump(model, MODEL_DIR / "market_model.joblib")
(MODEL_DIR / "market_encoders.json").write_text(json.dumps({
    "crop_map": crop_map,
    "market_map": market_map,
    "season_map": season_map,
}, indent=2))

print("Market model saved to:", MODEL_DIR / "market_model.joblib")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2  : {r2:.4f}")
