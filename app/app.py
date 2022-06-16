import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from src.preprocess import preprocess_pipeline
from src.inference import calculate_complexity

st.title("Complexité d'accompagnement")

data = pd.read_csv('data/student_data.csv')
df = preprocess_pipeline(data, predicting=True)
complexity = calculate_complexity(df, predicting=True)

fig, ax = plt.subplots()
ax.scatter(df['FinalGrade'], complexity['Complexity'], c=complexity['Complexity'])
ax.set_xlabel('Note finale')
ax.set_ylabel("Complexité d'accompagnement")

st.pyplot(fig, use_container_width=True)

