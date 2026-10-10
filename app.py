import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- HARDCODE YOUR THINGSPEAK CHANNEL HERE ---
DEFAULT_CHANNEL_ID = "3522166" 
DEFAULT_READ_KEY = "7F72LM8QZYKDBIYW"

# --- Page Config ---
st.set_page_config(page_title="EnerCast Analytics", layout="wide")

# --- Authentication Logic ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None

def login():
    st.title("EnerCast Portal Login")
    st.markdown("Please enter your access code to view the dashboard.")
    
    with st.form("login_form"):
        password = st.text_input("Access Code", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if password == "admin123":
                st.session_state['logged_in'] = True
                st.session_state['user_role'] = "Landlord"
                st.rerun()
            elif password == "boarder123":
                st.session_state['logged_in'] = True
                st.session_state['user_role'] = "Boarder"
                st.rerun()
            else:
                st.error("Invalid access code.")

def logout():
    st.session_state['logged_in'] = False
    st.session_state['user_role'] = None
    st.rerun()

if not st.session_state['logged_in']:
    login()
    st.stop() # Stop execution here if not logged in

# --- Main Dashboard ---
col_title, col_logout = st.columns([8, 1])
with col_title:
    st.title("EnerCast Analytics Dashboard")
    st.markdown(f"**View:** {st.session_state['user_role']} Mode")
with col_logout:
    st.button("Logout", on_click=logout)

st.divider()

# --- Sidebar Configuration (Landlord Only) ---
channel_id = DEFAULT_CHANNEL_ID
read_key = DEFAULT_READ_KEY
results_count = 500

if st.session_state['user_role'] == "Landlord":
    st.sidebar.header("Admin Configuration")
    channel_id = st.sidebar.text_input("Channel ID", value=DEFAULT_CHANNEL_ID)
    read_key = st.sidebar.text_input("Read API Key", value=DEFAULT_READ_KEY)
    results_count = st.sidebar.slider("Historical Data Points", min_value=50, max_value=8000, value=500)
    st.sidebar.info("Landlord Access: Full predictive analytics and configuration enabled.")
else:
    st.sidebar.info("Boarder Access: Viewing live power consumption and billing estimates.")

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
        df = df.rename(columns={
            'created_at': 'timestamp',
            'field1': 'current',
            'field2': 'power',
            'field3': 'energy'
        })
        
        # Convert types and adjust timezone from UTC to Philippine Standard Time (UTC+8)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Manila').dt.tz_localize(None)
        for col in ['current', 'power', 'energy']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna()
        return df
    except Exception as e:
        if st.session_state['user_role'] == "Landlord":
            st.sidebar.error(f"Error fetching data: {e}")
        return None

# Generate dummy data if no channel ID is provided
def generate_dummy_data():
    now_ph = datetime.utcnow() + timedelta(hours=8)
    dates = pd.date_range(end=now_ph, periods=500, freq='min')
    power = np.random.normal(50, 10, 500) 
    power[np.random.choice(500, 20)] += np.random.normal(500, 100, 20) 
    
    df = pd.DataFrame({
        'timestamp': dates,
        'current': power / 230.0,
        'power': power,
        'energy': np.cumsum(power / 60 / 1000) 
    })
    return df

# --- Fetch Data ---
df = fetch_thingspeak_data(channel_id, read_key, results_count)

if df is None:
    df = generate_dummy_data()

# --- Dashboard Layout (Visible to Everyone) ---
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

# --- Machine Learning (Visible ONLY to Landlord) ---
if st.session_state['user_role'] == "Landlord":
    st.divider()
    st.header("Energy Forecasting (Hybrid ARIMA-RF)")
    st.markdown("Training forecasting models based on historical data...")
    
    with st.spinner("Training models..."):
        ts_data = df.set_index('timestamp')['power'].resample('min').mean().interpolate()
        
        if len(ts_data) > 30:
            try:
                arima_model = ARIMA(ts_data, order=(1,1,1))
                arima_result = arima_model.fit()
                
                arima_fitted = arima_result.fittedvalues
                residuals = ts_data - arima_fitted
                
                rf_df = pd.DataFrame({'residual': residuals})
                for i in range(1, 4):
                    rf_df[f'lag_{i}'] = rf_df['residual'].shift(i)
                rf_df = rf_df.dropna()
                
                X = rf_df.drop('residual', axis=1)
                y = rf_df['residual']
                
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_model.fit(X, y)
                
                forecast_steps = 30
                arima_forecast = arima_result.forecast(steps=forecast_steps)
                
                future_residuals = []
                current_lags = y.iloc[-3:].values[::-1] 
                
                for _ in range(forecast_steps):
                    pred_res = rf_model.predict(current_lags.reshape(1, -1))[0]
                    future_residuals.append(pred_res)
                    current_lags = np.insert(current_lags, 0, pred_res)[:-1]
                    
                final_forecast = arima_forecast.values + np.array(future_residuals)
                final_forecast = np.clip(final_forecast, 0, None)
                
                last_date = ts_data.index[-1]
                future_dates = [last_date + timedelta(minutes=i) for i in range(1, forecast_steps + 1)]
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=ts_data.index[-100:], y=ts_data.values[-100:], mode='lines', name='Historical Data', line=dict(color='#64748b')))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=final_forecast, mode='lines', name='Hybrid Forecast', line=dict(color='#ef4444', dash='dash')))
                fig_pred.update_layout(height=400, title="Next 30 Minutes Power Forecast (ARIMA + RF)")
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error training models: {e}. Not enough data variance yet.")
        else:
            st.warning("Not enough historical data to train the AI model. Let the Arduino run for a few more minutes!")
