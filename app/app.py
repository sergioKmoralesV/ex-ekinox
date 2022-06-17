import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from src.preprocess import preprocess_pipeline
from src.inference import calculate_complexity

st.title("Complexité d'accompagnement")

st.subheader("Sélectionnez toutes les features à prendre en considération pour l'analyse")

col1, col2, col3 = st.columns(3)

with col1:
    st.session_state.famrel = st.checkbox('Relation familiale', value=True)
    st.session_state.Fedu = st.checkbox('Education du père', value=True)
    st.session_state.Fjob = st.checkbox('Travail du père', value=True)
    st.session_state.Medu = st.checkbox('Education de la mère', value=True)
    st.session_state.Mjob = st.checkbox('Travail de la mère', value=True)
    st.session_state.guardian = st.checkbox('Gardien', value=True)
    st.session_state.address = st.checkbox('Emplacement de son adresse ', value=True)
    st.session_state.internet = st.checkbox('Accès à Internet', value=True)

with col2:
    st.session_state.nursery = st.checkbox('École maternelle', value=True)
    st.session_state.higher = st.checkbox("Envie d'un niveau plus haut d'éducation", value=True)
    st.session_state.absences = st.checkbox("Nombre d'absences", value=True)
    st.session_state.failures = st.checkbox("Nombre de classes perdus", value=True)
    st.session_state.studytime = st.checkbox("Heures de revision par semaine", value=True)
    st.session_state.schoolsup = st.checkbox("Cours supplémentaires à l'école", value=True)
    st.session_state.paid = st.checkbox('Cours supplémentaires payantes', value=True)

with col3:
    st.session_state.activities = st.checkbox('Activités parascolaires', value=True)
    st.session_state.goout = st.checkbox('Sorties avec des amis', value=True)
    st.session_state.romantic = st.checkbox('Avec un rapport romantique', value=True)
    st.session_state.health = st.checkbox('État de santé', value=True)
    st.session_state.Dalc = st.checkbox("Consommation d'acool lors de la semaine", value=True)
    st.session_state.Walc = st.checkbox("Consommation d'acool en week-end", value=True)

data = pd.read_csv('data/student_data.csv')
df = preprocess_pipeline(data, predicting=True)
complexity = calculate_complexity(df, predicting=True, fields=st.session_state)

fig, ax = plt.subplots()
ax.scatter(df['FinalGrade'], complexity['Complexity'], c=complexity['Complexity'])
ax.set_xlabel('Note finale')
ax.set_ylabel("Complexité d'accompagnement")

st.pyplot(fig, use_container_width=True)

