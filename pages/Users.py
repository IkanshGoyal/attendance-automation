import streamlit as st
import pandas as pd

st.title("Attendance Web App")
st.subheader("Logged Users", divider = "grey")

column_names = ['Name', 'College ID', 'Class Roll Number', 'Section', 'Current Year']

df = pd.read_csv(
    'Users.csv',
    names=column_names
)

st.dataframe(df, use_container_width=True)