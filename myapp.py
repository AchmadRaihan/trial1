import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.write("""
# Simple Water Level Prediction App
This app predicts the **Water Level**.
""")

df1 = pd.read_csv('c.csv')
df2 = pd.read_csv('c.csv')

df2['ds'] = pd.to_datetime(df2['ds'])

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df2['ds'], y=df2['y'], name="stock_open"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

n_years = st.slider('Days of prediction:', 1, 37)
period = n_years * 24

m = Prophet().fit(df1)

future = m.make_future_dataframe(periods=period, freq='h')
forecast = m.predict(future)

st.write(f'Forecast plot for {n_years} days')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
