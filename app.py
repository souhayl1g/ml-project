import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your dataset
DATASET_PATH = r"C:\Users\Souhayl\Desktop\ML PROJECT\appendix.csv"

# Load dataset
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Train a simple RandomForest model
@st.cache_resource
def train_model(data, target_column):
    # Drop rows with missing values in the target column
    data = data.dropna(subset=[target_column])

    # Separate features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Ensure features are numeric
    X = X.select_dtypes(include=[np.number])

    # Ensure target is numeric
    y = pd.to_numeric(y, errors='coerce')

    # Drop rows with invalid target values
    valid_indices = y.notnull()
    X = X[valid_indices]
    y = y[valid_indices]

    # Check if there are enough samples
    if len(X) < 5 or len(y) < 5:  # Adjust minimum samples as needed
        raise ValueError("Dataset has too few valid samples after preprocessing.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)
    return model, X_test, y_test, predictions

# Generate recommendations (example: based on feature importance)
def generate_recommendations(model, feature_names):
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    recommendations = importance_df.sort_values(by='Importance', ascending=False).head(3)
    return recommendations

# Main App
def main():
    st.title("Machine Learning Project")
    st.sidebar.header("Navigation")
    action = st.sidebar.radio("Choose Action", ["Predict", "Profiling", "Recommendations"])

    # Load dataset
    data = load_data(DATASET_PATH)
    if data is None:
        return

    st.write("Dataset Preview:")
    st.write(data.head())

    # Handle actions
    if action == "Profiling":
        st.header("Data Profiling")
        st.write("Summary Statistics")
        st.write(data.describe())

        st.write("Correlation Matrix")
        numeric_data = data.select_dtypes(include=[np.number])  # Ensure only numeric columns are used
        if numeric_data.empty:
            st.warning("No numeric columns available for correlation matrix.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    elif action == "Predict":
        st.header("Prediction")
        target_column = st.selectbox("Select Target Column", data.columns)
        if st.button("Train and Predict"):
            try:
                model, X_test, y_test, predictions = train_model(data, target_column)
                st.write("R² Score:", r2_score(y_test, predictions))
                st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))

                st.write("Sample Predictions:")
                result = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
                st.write(result.head())
            except ValueError as e:
                st.error(f"Error during training: {e}")

    elif action == "Recommendations":
        st.header("Recommendations")
        target_column = st.selectbox("Select Target Column for Recommendations", data.columns)
        if st.button("Generate Recommendations"):
            try:
                model, _, _, _ = train_model(data, target_column)
                recommendations = generate_recommendations(model, data.drop(columns=[target_column]).select_dtypes(include=[np.number]).columns)
                st.write("Top Feature Recommendations:")
                st.write(recommendations)
            except ValueError as e:
                st.error(f"Error during recommendation generation: {e}")

if __name__ == "__main__":
    main()
