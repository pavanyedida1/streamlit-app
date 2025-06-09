import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

st.title("🌸 Iris Flower Classifier with Visualization")
st.write("Use the sliders to input flower measurements and see predictions with charts.")

# User input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction
prediction = model.predict(input_data)[0]
probs = model.predict_proba(input_data)[0]

predicted_class = target_names[prediction]

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Result")
    st.success(f"🌼 Predicted Species: **{predicted_class.capitalize()}**")
    st.write("Prediction Probabilities:")
    prob_df = pd.DataFrame({
        'Species': target_names,
        'Probability': probs
    }).set_index('Species')
    st.bar_chart(prob_df)

with col2:
    st.subheader("Scatter Plot (Petal Length vs Petal Width)")

    fig, ax = plt.subplots()
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        idx = y == i
        ax.scatter(X[idx, 2], X[idx, 3], c=color, label=target_names[i], alpha=0.5)
    # Plot user input point
    ax.scatter(petal_length, petal_width, c='black', label='Your Input', marker='X', s=100)

    ax.set_xlabel('Petal Length (cm)')
    ax.set_ylabel('Petal Width (cm)')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# Optional: Show raw data
with st.expander("Show raw input data"):
    st.write({
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    })
