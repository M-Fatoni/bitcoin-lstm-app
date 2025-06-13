#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

# ========== CONFIG ========== #
st.set_page_config(page_title="Bitcoin Price Forecast", layout="wide")
st.title("📈 Bitcoin Price Forecasting App")
st.markdown("Aplikasi ini memprediksi harga Bitcoin berdasarkan data historis menggunakan model LSTM.")

# ========== INPUT ========== #
n_days = st.slider("🔧 Berapa hari ke depan yang ingin Anda prediksi?", min_value=1, max_value=30, value=7)

# ========== LOAD DATA ========== #
@st.cache_data
def load_data():
    url = 'https://docs.google.com/spreadsheets/d/1g-MlWh_MjuIUaKq7u7YUO_uTewP5369DpYUY33hb2uc/export?format=csv'
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df[['Close']]

data = load_data()

# ========== TAMPILKAN DATA HISTORIS ========== #
st.subheader("📊 Data Historis Harga Bitcoin")

hist_fig = go.Figure()

hist_fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='Harga Penutupan',
    line=dict(color='green'),
    hovertemplate='Tanggal: %{x|%Y-%m-%d}<br>Harga: $%{y:,.2f}<extra></extra>'
))

hist_fig.update_layout(
    title='Trend Historis Harga Bitcoin',
    xaxis_title='Tanggal',
    yaxis_title='Harga (USD)',
    template='plotly_white',
    hovermode='x unified',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=7, label='1W', step='day', stepmode='backward'),
                dict(count=1, label='1M', step='month', stepmode='backward'),
                dict(count=3, label='3M', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible=True),
        type='date'
    )
)

st.plotly_chart(hist_fig, use_container_width=True)

# ========== PREPROCESSING ========== #
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

past_60 = scaled_data[-60:]
X_input = np.array(past_60).reshape(1, 60, 1)

# ========== LOAD MODEL ========== #
model = load_model("bitcoin_lstm_model.h5")

# ========== PREDIKSI ========== #
predictions = []
input_seq = X_input.copy()

for _ in range(n_days):
    pred = model.predict(input_seq, verbose=0)[0][0]
    predictions.append(pred)
    input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
last_price = data['Close'][-1]
future_dates = [data.index[-1] + datetime.timedelta(days=i+1) for i in range(n_days)]

# ========== BUAT DATAFRAME HASIL PREDIKSI ========== #
pred_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price': predicted_prices.flatten()
})
pred_df.set_index('Date', inplace=True)

# ========== SCORECARD ========== #
delta = predicted_prices[-1][0] - last_price
delta_str = f"{delta:.2f}"
delta_color = "normal"
if delta > 0:
    delta_color = "inverse"
elif delta < 0:
    delta_color = "off"

col1, col2 = st.columns(2)
with col1:
    st.metric(label=f"Harga Bitcoin Saat Ini", value=f"${last_price:,.2f}")
with col2:
    st.metric(label=f"Prediksi Hari ke-{n_days}", value=f"${predicted_prices[-1][0]:,.2f}", delta=delta_str)

# ========== VISUALISASI DENGAN PLOTLY ========== #
combined_df = pd.concat([data.tail(30), pred_df])

fig = go.Figure()
fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['Close'], 
                         mode='lines', name='Harga Aktual', 
                         line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted Price'], 
                         mode='lines+markers', name='Prediksi', 
                         line=dict(color='orange', dash='dash')))
fig.update_layout(title="🔮 Visualisasi Prediksi Harga Bitcoin", 
                  xaxis_title="Tanggal", yaxis_title="Harga (USD)", 
                  hovermode="x unified",
                  template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# ========== DATA PREDIKSI ========== #
with st.expander("Lihat Tabel Data Prediksi"):
    st.dataframe(pred_df.style.format("${:.2f}"))

st.success("✅ Prediksi selesai. Ubah jumlah hari untuk melihat skenario berbeda.")
