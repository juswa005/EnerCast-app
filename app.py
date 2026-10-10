import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="EnerCast Analytics", layout="wide")
st.title("EnerCast Analytics Dashboard")
st.markdown("IoT Energy Monitoring & Forecasting using Hybrid ARIMA-RF")

# --- Sidebar Configuration ---
st.sidebar.header("ThingSpeak Configuration")
channel_id = st.sidebar.text_input("Channel ID", value="")
read_key = st.sidebar.text_input("Read API Key (if private)", value="")
results_count = st.sidebar.slider("Number of Data Points", min_value=50, max_value=8000, value=500)

@st.cache_data(ttl=60) # Cache data for 60 seconds
def fetch_thingspeak_data(channel_id, read_key, results):
    if not channel_id:
        return None
        
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?results={results}"
    if read_key:
        url += f"&api_key={read_key}"
        
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'feeds' not in data:
            return None
            
        df = pd.DataFrame(data['feeds'])
        # Rename columns based on our Arduino code mapping
        # field1 = Voltage, field2 = Current, field3 = Power, field4 = Energy
        df = df.rename(columns={
            'created_at': 'timestamp',
            'field1': 'voltage',
            'field2': 'current',
            'field3': 'power',
            'field4': 'energy'
        })
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['voltage', 'current', 'power', 'energy']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna()
        return df
    except Exception as e:
        st.sidebar.error(f"Error fetching data: {e}")
        return None

# Generate dummy data if no channel ID is provided
def generate_dummy_data():
    dates = pd.date_range(end=datetime.now(), periods=500, freq='min')
    # Generate realistic looking power consumption (baseline + spikes)
    power = np.random.normal(50, 10, 500) # Baseline 50W
    power[np.random.choice(500, 20)] += np.random.normal(500, 100, 20) # Add spikes (e.g. fridge turning on)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'voltage': 230.0,
        'current': power / 230.0,
        'power': power,
        'energy': np.cumsum(power / 60 / 1000) # Roughly accumulating kWh
    })
    return df

# --- Fetch Data ---
df = fetch_thingspeak_data(channel_id, read_key, results_count)

if df is None:
    df = generate_dummy_data()

# --- Dashboard Layout ---
col1, col2, col3, col4 = st.columns(4)
latest = df.iloc[-1]
col1.metric("Live Power", f"{latest['power']:.2f} W")
col2.metric("Live Current", f"{latest['current']:.2f} A")
col3.metric("Total Energy", f"{latest['energy']:.4f} kWh")
cost_per_kwh = 11.0 # Average PHP cost per kWh
col4.metric("Estimated Cost", f"₱ {latest['energy'] * cost_per_kwh:.2f}")

st.subheader("Live Power Consumption")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['power'], mode='lines', name='Power (W)', line=dict(color='#2563eb')))
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
st.plotly_chart(fig, use_container_width=True)

# --- Machine Learning: ARIMA + Random Forest Hybrid ---
st.header("Energy Forecasting (Hybrid ARIMA-RF)")
st.markdown("Training forecasting models based on historical data...")

with st.spinner("Training models..."):
    # Prepare time series data (resample to avoid missing gaps)
    ts_data = df.set_index('timestamp')['power'].resample('min').mean().interpolate()
    
    # We need at least a few points to train
    if len(ts_data) > 30:
        # 1. ARIMA Model
        # Simple configuration (p,d,q) = (1,1,1) for fast execution
        try:
            arima_model = ARIMA(ts_data, order=(1,1,1))
            arima_result = arima_model.fit()
            
            # Get ARIMA fitted values and calculate residuals
            arima_fitted = arima_result.fittedvalues
            residuals = ts_data - arima_fitted
            
            # 2. Random Forest for Residuals (Nonlinear component)
            # Create lagged features for the RF model
            rf_df = pd.DataFrame({'residual': residuals})
            for i in range(1, 4):
                rf_df[f'lag_{i}'] = rf_df['residual'].shift(i)
            rf_df = rf_df.dropna()
            
            X = rf_df.drop('residual', axis=1)
            y = rf_df['residual']
            
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_model.fit(X, y)
            
            # 3. Forecast Future (Next 30 periods/minutes)
            forecast_steps = 30
            arima_forecast = arima_result.forecast(steps=forecast_steps)
            
            # Predict future residuals
            future_residuals = []
            current_lags = y.iloc[-3:].values[::-1] # Last 3 residuals
            
            for _ in range(forecast_steps):
                pred_res = rf_model.predict(current_lags.reshape(1, -1))[0]
                future_residuals.append(pred_res)
                # Update lags
                current_lags = np.insert(current_lags, 0, pred_res)[:-1]
                
            # Combine ARIMA forecast + RF predicted residuals
            final_forecast = arima_forecast.values + np.array(future_residuals)
            
            # Ensure no negative power predictions
            final_forecast = np.clip(final_forecast, 0, None)
            
            # Create Future Dates
            last_date = ts_data.index[-1]
            future_dates = [last_date + timedelta(minutes=i) for i in range(1, forecast_steps + 1)]
            
            # Plot Results
            fig_pred = go.Figure()
            # Historical
            fig_pred.add_trace(go.Scatter(x=ts_data.index[-100:], y=ts_data.values[-100:], mode='lines', name='Historical Data', line=dict(color='#64748b')))
            # Forecast
            fig_pred.add_trace(go.Scatter(x=future_dates, y=final_forecast, mode='lines', name='Hybrid Forecast', line=dict(color='#ef4444', dash='dash')))
            fig_pred.update_layout(height=400, title="Next 30 Minutes Power Forecast (ARIMA + RF)")
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error training models: {e}. Not enough data variance yet.")
    else:
        st.warning("Not enough historical data to train the AI model. Let the Arduino run for a few more minutes!")
