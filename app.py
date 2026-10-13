import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

from forecasting import MODELS, backtest, forecast, prepare_series


def secret(key, default):
    """Read a value from .streamlit/secrets.toml, falling back to the default."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# --- HARDCODE YOUR THINGSPEAK CHANNEL HERE (or set it in .streamlit/secrets.toml) ---
DEFAULT_CHANNEL_ID = secret("channel_id", "3522166")
DEFAULT_READ_KEY = secret("read_key", "7F72LM8QZYKDBIYW")
DEFAULT_RATE = float(secret("rate_per_kwh", 11.0))  # ₱ per kWh
# Show "simulated" tags/banners on rooms without live data (off for presentations)
LABEL_SIMULATED = bool(secret("label_simulated_data", False))

# Access code -> (role, room)
ACCESS_CODES = {
    secret("landlord_code", "admin123"): ("Landlord", None),
    secret("room1_code", "boarder123"): ("Boarder", "Room 1"),
    secret("room2_code", "boarder456"): ("Boarder", "Room 2"),
}

TIMEZONE = 'Asia/Manila'
GAP = pd.Timedelta(minutes=5)          # no reading for this long = device offline
FORECAST_STEPS = 30

# --- Page Config ---
st.set_page_config(page_title="EnerCast Analytics", layout="wide")

# --- Authentication Logic ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_role' not in st.session_state:
    st.session_state['user_role'] = None
if 'tenant_room' not in st.session_state:
    st.session_state['tenant_room'] = None

def login():
    st.title("EnerCast Portal Login")
    st.markdown("Please enter your access code to view the dashboard.")

    with st.form("login_form"):
        password = st.text_input("Access Code", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if password in ACCESS_CODES:
                role, room = ACCESS_CODES[password]
                st.session_state['logged_in'] = True
                st.session_state['user_role'] = role
                st.session_state['tenant_room'] = room
                st.rerun()
            else:
                st.error("Invalid access code.")

def logout():
    st.session_state['logged_in'] = False
    st.session_state['user_role'] = None
    st.session_state['tenant_room'] = None

if not st.session_state['logged_in']:
    login()
    st.stop()

is_landlord = st.session_state['user_role'] == "Landlord"

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
rate = DEFAULT_RATE
model_choice = MODELS[0]

if is_landlord:
    st.sidebar.header("Admin Configuration")
    channel_id = st.sidebar.text_input("Room 1 Channel ID", value=DEFAULT_CHANNEL_ID)
    read_key = st.sidebar.text_input("Room 1 Read API Key", value=DEFAULT_READ_KEY, type="password")
    results_count = st.sidebar.slider("Historical Data Points", min_value=50, max_value=8000, value=500,
                                      help="ThingSpeak readings arrive every ~20 s, so 500 points ≈ 2.8 hours.")
    rate = st.sidebar.number_input("Electricity Rate (₱/kWh)", min_value=0.0, value=DEFAULT_RATE, step=0.5)
    model_choice = st.sidebar.selectbox("Forecast Model", MODELS,
                                        help="Use the accuracy check under each forecast to see which model fits your data best.")
    st.sidebar.info("Landlord Access: Viewing data and AI forecasts for ALL rooms.")
else:
    st.sidebar.info(f"Boarder Access: You are only viewing data for {st.session_state['tenant_room']}.")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_thingspeak_data(channel_id, read_key, results):
    """Returns (DataFrame or None, error message or None)."""
    if not channel_id:
        return None, "No channel ID configured."
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
    params = {'results': results}
    if read_key:
        params['api_key'] = read_key
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return None, f"Could not reach ThingSpeak ({e.__class__.__name__})."
    if not isinstance(data, dict) or not data.get('feeds'):
        return None, "ThingSpeak returned no readings (check the channel ID / read key)."

    df = pd.DataFrame(data['feeds'])
    df = df.rename(columns={
        'created_at': 'timestamp',
        'field1': 'current',
        'field2': 'power',
        'field3': 'energy'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df['timestamp'] = df['timestamp'].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    for col in ['current', 'power', 'energy']:
        df[col] = pd.to_numeric(df.get(col), errors='coerce')

    # Older firmware uploaded field1=Voltage, field2=Current, field3=Power.
    # Those rows have field1 ≈ 230 (impossible as a current on a 20 A sensor) and
    # field3 ≈ field1 × field2. Re-map them instead of reading 230 V as 230 A.
    legacy = (df['current'] >= 100) & ((df['energy'] - df['current'] * df['power']).abs()
                                       <= 0.05 * df['energy'].abs() + 1)
    if legacy.any():
        df.loc[legacy, 'current'] = df.loc[legacy, 'power']
        df.loc[legacy, 'power'] = df.loc[legacy, 'energy']
        df.loc[legacy, 'energy'] = np.nan

    df = df[['timestamp', 'current', 'power', 'energy']].dropna(subset=['current', 'power'])
    df = df.sort_values('timestamp')
    if df.empty:
        return None, "ThingSpeak readings are empty."
    return df.reset_index(drop=True), None


def now_ph():
    return pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)


def generate_simulated_room(seed, base_power):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=now_ph().floor('min'), periods=500, freq='min')
    power = rng.normal(base_power, 10, 500)
    power[rng.choice(500, 15)] += rng.normal(300, 50, 15)
    power = np.clip(power, 0, None)
    df = pd.DataFrame({
        'timestamp': dates,
        'current': power / 230.0,
        'power': power,
        'energy': np.cumsum(power / 60 / 1000)
    })
    return df


# Fetch Live Data for Room 1
df_room1, room1_error = fetch_thingspeak_data(channel_id, read_key, results_count)
room1_live = df_room1 is not None
if not room1_live:
    df_room1 = generate_simulated_room(1, 50)  # Fallback

# room -> (data, is_live, note)
ROOMS = {
    "Room 1": (df_room1, room1_live, room1_error),
    "Room 2": (generate_simulated_room(42, 60), False, None),
    "Room 3": (generate_simulated_room(99, 45), False, None),
}


def energy_used(energy):
    """kWh used over the period, tolerant of the device's counter being reset."""
    energy = energy.dropna()  # rows from old firmware have no energy reading
    d = energy.diff()
    # A drop means the meter restarted from 0; the new reading is what was used since.
    return float(d.clip(lower=0).sum() + energy[d < 0].sum())


def hourly_energy(df):
    """kWh per hour, estimated by integrating power. Offline gaps count as 0."""
    dt_h = df['timestamp'].diff().shift(-1)
    dt_h = dt_h.where(dt_h <= GAP).dt.total_seconds().fillna(0) / 3600
    kwh = df['power'] * dt_h / 1000
    return kwh.groupby(df['timestamp'].dt.floor('h')).sum()


def with_gaps(df):
    """Insert empty rows at offline gaps so charts don't draw lines across them."""
    gaps = df['timestamp'].diff() > GAP
    if not gaps.any():
        return df
    breaks = pd.DataFrame({'timestamp': df.loc[gaps, 'timestamp'] - pd.Timedelta(seconds=1)})
    return pd.concat([df, breaks]).sort_values('timestamp')


def render_room_metrics(room, df_room, is_live, note):
    if not is_live and LABEL_SIMULATED:
        if note:
            st.warning(f"Showing **simulated** data — live data unavailable: {note}")
        else:
            st.info("This room has no meter installed yet — showing **simulated** demo data.")
    else:
        age = now_ph() - df_room['timestamp'].iloc[-1]
        if age > GAP:
            st.warning(f"Device may be offline — last reading was {int(age.total_seconds() // 60)} min ago "
                       f"({df_room['timestamp'].iloc[-1]:%b %d, %I:%M %p}).")

    latest = df_room.iloc[-1]
    meter = df_room['energy'].dropna()
    meter = float(meter.iloc[-1]) if len(meter) else 0.0
    used = energy_used(df_room['energy'])
    hours = (df_room['timestamp'].iloc[-1] - df_room['timestamp'].iloc[0]).total_seconds() / 3600
    avg_power = df_room['power'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Live Power", f"{latest['power']:.2f} W")
    col2.metric("Live Current", f"{latest['current']:.2f} A")
    col3.metric("Meter Reading", f"{meter:.4f} kWh",
                help="Total energy counted by the device.")
    col4.metric("Estimated Cost", f"₱ {meter * rate:.2f}",
                help=f"Meter reading × ₱{rate:.2f}/kWh")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Used This Period", f"{used:.4f} kWh", help=f"Over the last {hours:.1f} hours shown.")
    col6.metric("Cost This Period", f"₱ {used * rate:.2f}")
    col7.metric("Average Power", f"{avg_power:.1f} W")
    col8.metric("Projected Monthly Cost", f"₱ {avg_power * 24 * 30 / 1000 * rate:,.2f}",
                help="If the average power above continued for 30 days.")

    plot_df = with_gaps(df_room)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['timestamp'], y=plot_df['power'], mode='lines', name='Power (W)', line=dict(color='#2563eb')))
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300, title="Power (W)", yaxis_title="W")
    st.plotly_chart(fig, width='stretch')

    hourly = hourly_energy(df_room)
    if len(hourly) > 1:
        fig_h = go.Figure(go.Bar(x=hourly.index, y=hourly.values, marker_color='#0ea5e9', name='kWh',
                                 hovertemplate='%{x|%b %d, %I %p}<br>%{y:.4f} kWh<extra></extra>'))
        fig_h.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=250, title="Hourly Consumption (kWh)", yaxis_title="kWh")
        st.plotly_chart(fig_h, width='stretch')

    st.download_button("Download data (CSV)", df_room.to_csv(index=False).encode('utf-8'),
                       file_name=f"enercast_{room.replace(' ', '_').lower()}.csv", mime="text/csv",
                       key=f"dl_{room}")


@st.cache_data(ttl=300, show_spinner=False)
def run_forecast(ts, model):
    return forecast(ts, model=model, steps=FORECAST_STEPS)


@st.cache_data(ttl=300, show_spinner=False)
def run_backtest(ts):
    return backtest(ts, steps=FORECAST_STEPS)


def render_ml_forecast(room, df_room):
    st.divider()
    st.subheader(f"Energy Forecasting ({model_choice})")

    ts_data = prepare_series(df_room)
    if len(ts_data) < 60:
        st.warning("Not enough continuous data to train the AI model yet (need at least 60 minutes).")
        return

    with st.spinner("Training models..."):
        try:
            pred, description = run_forecast(ts_data, model_choice)
        except Exception as e:
            st.error(f"Error training models: {e}")
            return

    next_kwh = pred['forecast'].sum() / 60 / 1000
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Energy (next 30 min)", f"{next_kwh:.4f} kWh")
    c2.metric("Expected Cost (next 30 min)", f"₱ {next_kwh * rate:.2f}")
    c3.metric("Expected Peak", f"{pred['forecast'].max():.1f} W")

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=ts_data.index[-100:], y=ts_data.values[-100:], mode='lines', name='Historical Data', line=dict(color='#64748b')))
    if pred['upper'].notna().any():
        fig_pred.add_trace(go.Scatter(x=pred.index, y=pred['upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_pred.add_trace(go.Scatter(x=pred.index, y=pred['lower'], mode='lines', line=dict(width=0), fill='tonexty',
                                      fillcolor='rgba(239,68,68,0.15)', name='80% Range', hoverinfo='skip'))
    fig_pred.add_trace(go.Scatter(x=pred.index, y=pred['forecast'], mode='lines', name='Forecast', line=dict(color='#ef4444', dash='dash')))
    fig_pred.update_layout(height=400, title="Next 30 Minutes Power Forecast", yaxis_title="W")
    st.plotly_chart(fig_pred, width='stretch')
    st.caption(f"Model: {description}. Trained on the last {len(ts_data)} minutes of continuous data.")

    if st.toggle("Check model accuracy", key=f"bt_{room}",
                 help="Hides the last 30 minutes, forecasts them with each model, and compares against what really happened."):
        with st.spinner("Evaluating models..."):
            try:
                scores = run_backtest(ts_data)
            except ValueError as e:
                st.info(f"Can't check accuracy yet: {e}")
                return
            except Exception as e:
                st.error(f"Could not evaluate models: {e}")
                return
        best = scores['RMSE (W)'].idxmin()
        st.dataframe(scores.style.format("{:.3f}").highlight_min(subset=['RMSE (W)', 'MAE (W)'], color='#bbf7d0')
                     .highlight_max(subset=['R²'], color='#bbf7d0'), width='stretch')
        st.caption(f"Lowest error on the held-out 30 minutes: **{best}**. Results from a single 30-minute "
                   "window vary — R² can be negative when power was nearly flat during the test.")


# --- Render Views ---
if not is_landlord:
    room = st.session_state['tenant_room']
    df_room, is_live, note = ROOMS[room]
    render_room_metrics(room, df_room, is_live, note)
else:
    # Landlord sees all rooms in tabs
    tabs = st.tabs([name if ROOMS[name][1] or not LABEL_SIMULATED else f"{name} (simulated)" for name in ROOMS])
    for tab, (room, (df_room, is_live, note)) in zip(tabs, ROOMS.items()):
        with tab:
            render_room_metrics(room, df_room, is_live, note)
            render_ml_forecast(room, df_room)
