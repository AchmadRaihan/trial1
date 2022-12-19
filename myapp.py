import streamlit as st
import pandas as pd
from prophet import Prophet
import datetime
import altair as alt
import plotly.express as px

st.write("""
# Simple Water Level Prediction App
This app predicts the **Water Level**.
""")

df1 = pd.read_csv('c.csv')
df2 = pd.read_csv('c.csv')

df2['ds'] = pd.to_datetime(df2['ds'])

m = Prophet().fit(df1)

future = m.make_future_dataframe(periods=901, freq='h')
forecast = m.predict(future)

a = alt.Chart(df2).mark_line().encode(
    x='ds',
    y='y',
    color=alt.value('red')
).interactive()
b = alt.Chart(forecast).mark_line().encode(
    x='ds',
    y='yhat',
    color=alt.value('blue')
).interactive()

a + b