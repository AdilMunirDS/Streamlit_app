import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Load your data
@st.cache_data
def load_data():
    df1 = pd.read_excel('cleaned_data_with_features.xlsx', sheet_name=1)
    df2 = pd.read_excel('cleaned_data_with_features.xlsx', sheet_name=2)
    df2.columns = df1.columns = ["Sr. No.", "ID", "Age", "Gender", "A-500", "R-500", "N-500", "T-500", "P-500"]
    df = pd.concat([df1, df2], ignore_index=True)
    return df

df = load_data()

# Normalize the data
columns_to_scale = ["Age", "A-500", "R-500", "N-500", "T-500"]
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Prepare features and target
X = df[["Age", "Gender", "A-500", "R-500", "N-500", "T-500"]]
y = df["P-500"]

# Convert categorical variable
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
X.to_excel('output.xlsx', index=False)

# Streamlit app
st.title("PTA Threshold Prediction App")

# Input features with default values
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"], index=0)  # Default to "Male"
a_500 = st.number_input("A-500", value=50)
r_500 = st.number_input("R-500", value=20)
n_500 = st.number_input("N-500", value=10)
t_500 = st.number_input("T-500", value=1500)

# Model selection
model_option = st.selectbox("Select a model", ["LinearRegression", "SVM", "RandomForest", "DecisionTree", "KNN"])

# Hyperparameter tuning options
params = {}
if model_option == "SVM":
    params = {
        'C': [st.slider("C (Regularization)", 0.01, 10.0, 1.0)],  # Wrap in a list
        'kernel': [st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])]
    }
elif model_option == "RandomForest":
    params = {
        'n_estimators': [st.slider("Number of Trees", 10, 200, 100)],
        'max_depth': [st.slider("Max Depth", 1, 20, 10)]
    }
elif model_option == "DecisionTree":
    params = {
        'max_depth': [st.slider("Max Depth", 1, 20, 10)]
    }
elif model_option == "KNN":
    n_neighbors = st.slider("Number of Neighbors", 1, 30, 5)
    params = {
        'n_neighbors': [n_neighbors]  # Wrap in a list
    }

# Prepare input data for prediction
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "A-500": [a_500],
    "R-500": [r_500],
    "N-500": [n_500],
    "T-500": [t_500]
})

# Normalize input data
input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])
input_data = pd.get_dummies(input_data, columns=['Gender'], drop_first=True).reindex(columns=X.columns, fill_value=0)

# Button to train and predict
if st.button("Predict"):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the selected model
    if model_option == "LinearRegression":
        model = LinearRegression()
    elif model_option == "SVM":
        model = SVR()
    elif model_option == "RandomForest":
        model = RandomForestRegressor()
    elif model_option == "DecisionTree":
        model = DecisionTreeRegressor()
    elif model_option == "KNN":
        model = KNeighborsRegressor()

    # Perform hyperparameter tuning if applicable
    if params:
        grid_search = GridSearchCV(model, param_grid=params, cv=5)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(input_data)

    # Display prediction result
    st.write(f"Predicted PTA Threshold (P-500): {y_pred[0]:.2f}")

    # Calculate and display model performance metrics
    y_test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    st.write(f"Mean Absolute Error: {mae:.2f}")

# Run the app with:
# streamlit run app.py
