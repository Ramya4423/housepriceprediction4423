import os
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Page settings
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Step 2: Load dataset
file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "house_data.csv"
)

if not os.path.exists(file_path):
    st.error("house_data.csv file not found!")
    st.info("Keep house_data.csv in the same folder as this Python file.")
    st.stop()

data = pd.read_csv(file_path)

# Step 3: Check required columns
required_columns = [
    "Area", "Bedrooms", "Bathrooms", "Floors",
    "YearBuilt", "Location", "Condition", "Garage", "Price"
]

missing = [col for col in required_columns if col not in data.columns]

if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

data = data.dropna(subset=required_columns)

if data.empty:
    st.error("Dataset has no complete rows.")
    st.stop()

# Step 4: Prepare features and target
X = data[[
    "Area", "Bedrooms", "Bathrooms", "Floors",
    "YearBuilt", "Location", "Condition", "Garage"
]]

y = data["Price"]

numeric_columns = [
    "Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"
]

categorical_columns = ["Location", "Condition", "Garage"]

# Step 5: Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_columns)
    ]
)

# Step 6: Train the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Step 7: Model evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 8: Website interface
st.title("🏠 House Price Prediction")
st.write("Enter house details to estimate the price.")

st.subheader("House Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (sq ft)", min_value=1, value=1500
    )

    bedrooms = st.number_input(
        "Bedrooms", min_value=1, max_value=20, value=3
    )

    bathrooms = st.number_input(
        "Bathrooms", min_value=1, max_value=20, value=2
    )

    floors = st.number_input(
        "Floors", min_value=1, max_value=10, value=2
    )

with col2:
    year = st.number_input(
        "Year Built", min_value=1800, max_value=2026, value=2010
    )

    locations = sorted(data["Location"].astype(str).unique())
    location = st.selectbox("Location", locations)

    conditions = sorted(data["Condition"].astype(str).unique())
    condition = st.selectbox("Condition", conditions)

    garages = sorted(data["Garage"].astype(str).unique())
    garage = st.selectbox("Garage", garages)

# Step 9: Predict house price
if st.button("Predict Price", type="primary"):
    input_data = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "YearBuilt": year,
        "Location": location,
        "Condition": condition,
        "Garage": garage
    }])

    predicted_price = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ₹{predicted_price:,.2f}")

    st.caption(
        "This is an estimate based on the available training dataset, "
        "not a guaranteed market price."
    )

# Step 10: Display model evaluation
with st.expander("Model Performance"):
    st.write(f"Mean Squared Error (MSE): {mse:,.2f}")
    st.write(f"R² Score: {r2:.4f}")