import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("insurance.csv")

# Model Training
X = df[['age']]
y = df['bought_insurance']
model = LogisticRegression()
model.fit(X, y)

# Streamlit App
st.title("Insurance Purchase Prediction")
st.write("This app predicts whether a person will buy insurance based on their age using a logistic regression model.")

# User input for age
age_input = st.number_input("Enter age:", min_value=0, max_value=100, value=25)

# Prediction
if st.button("Predict"):
    prediction = model.predict([[age_input]])[0]
    if prediction == 1:
        st.success("The person is likely to buy insurance.")
    else:
        st.warning("The person is not likely to buy insurance.")

# Display dataset
st.subheader("Dataset")
st.write(df)

# Data Visualization
st.subheader("Data Visualization")
fig, ax = plt.subplots()
ax.scatter(df['age'], df['bought_insurance'])
ax.set_xlabel("Age")
ax.set_ylabel("Bought Insurance")
st.pyplot(fig)
