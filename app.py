import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction System")
st.write("Predict house prices using your actual dataset.")

# Load your actual CSV dataset
data_path = Path(__file__).parent / "house_data.csv"

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error("house_data.csv file not found. Keep it in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Unable to read dataset: {e}")
    st.stop()

# Clean column names
df.columns = df.columns.str.strip()

required_columns = [
    "Area", "Bedrooms", "Bathrooms", "Floors",
    "YearBuilt", "Location", "Condition", "Garage", "Price"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

# Prepare dataset
df = df[required_columns].copy()
numeric_cols = [
    "Area", "Bedrooms", "Bathrooms",
    "Floors", "YearBuilt", "Price"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Price"])
df = df[df["Price"] > 0]

if len(df) < 10:
    st.error("Dataset has too few valid rows for training.")
    st.stop()

X = df.drop(columns=["Price"])
y = df["Price"]

numeric_features = [
    "Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"
]
categorical_features = ["Location", "Condition", "Garage"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

test_predictions = model.predict(X_test)

mse = mean_squared_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

st.success(f"Dataset loaded successfully: {len(df)} records")

with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)

st.subheader("Enter House Details")

col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input(
        "Area (sq ft)", min_value=1, max_value=100000, value=2000
    )
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])

with col2:
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
    floors = st.selectbox("Floors", [1, 2, 3])

with col3:
    year_built = st.number_input(
        "Year Built", min_value=1800, max_value=2100, value=2000
    )
    garage = st.selectbox("Garage", ["Yes", "No"])

location = st.selectbox(
    "Location",
    sorted(df["Location"].dropna().astype(str).unique())
)

condition = st.selectbox(
    "Condition",
    sorted(df["Condition"].dropna().astype(str).unique())
)

if st.button("Predict House Price", type="primary"):
    input_data = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "YearBuilt": year_built,
        "Location": location,
        "Condition": condition,
        "Garage": garage
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.metric("Estimated House Price", f"{prediction:,.2f}")

    st.caption(
        "This is a model estimate based on the supplied dataset, "
        "not a guaranteed market price."
    )

st.subheader("Model Performance on Test Data")

m1, m2 = st.columns(2)
m1.metric("Mean Squared Error", f"{mse:,.2f}")
m2.metric("R² Score", f"{r2:.4f}")

st.caption(
    "Performance depends on the quality and consistency of the dataset."
)