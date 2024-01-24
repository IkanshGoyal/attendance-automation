import streamlit as st
import pandas as pd

st.title("Attendance Web App")
st.subheader("Attendance", divider = "gray")

column_names = ['Name', 'College id', 'Login Time', 'Date']

df = pd.read_csv(
    'Attendance.csv',
    names=column_names
)

st.dataframe(df, use_container_width=True)