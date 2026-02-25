import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- 1. UI SETTINGS ---
st.set_page_config(page_title="Smart Store AI", layout="wide", initial_sidebar_state="expanded")
st.title("₹ Profit & Inventory Optimizer")

# --- INTERACTIVE SIDEBAR FOR MULTIPLE OWNERS ---
with st.sidebar:
    st.header("🏪 Store Settings")
    owner_name = st.text_input("Shop Owner Name", "Gupta Electronics")
    item_name = st.text_input("Target Item", "Smartphones")
    item_price = st.number_input("Avg Price per Item (₹)", value=15000, step=500)
    
    st.markdown("---")
    st.header("⚙️ 'What-If' Optimizer")
    st.caption("Play with these sliders to see how your profit changes.")
    risk_tolerance = st.slider("Stock-out Risk Tolerance", 1, 10, 5, help="1 = High Safety Stock, 10 = Lean Stock")

# --- 2. DATA GENERATION (Simulated for Demo) ---
@st.cache_data
def load_data():
    dates = pd.date_range(end=datetime.now(), periods=30)
    sales = np.random.randint(5, 20, size=30)
    # Add a random spike to show how AI catches it
    sales[20:23] += 25
    return pd.DataFrame({'Date': dates, 'Sales': sales})

df = load_data()

# --- 3. THE ALGORITHM BLOCK (FUTURE-PROOF) ---
# You can swap this whole function out in the future without breaking the UI
def run_prediction_algo(data, risk_factor):
    data['Day'] = data['Date'].dt.dayofweek
    X = data[['Day']]
    y = data['Sales']
    
    # Current Algo: Random Forest
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    data['Forecast'] = model.predict(X)
    data['Residual'] = data['Sales'] - data['Forecast']
    
    # Optimization Math: Lower risk tolerance = higher safety buffer
    safety_multiplier = (11 - risk_factor) / 5.0 
    safety_buffer = np.maximum(0, np.percentile(data['Residual'], 85) * safety_multiplier)
    
    data['Optimized_Stock'] = data['Forecast'] + safety_buffer
    return data, safety_buffer

df, current_safety_buffer = run_prediction_algo(df, risk_tolerance)

# --- 4. DASHBOARD & ₹ METRICS ---
st.markdown(f"### Welcome, {owner_name} | Analyzing: **{item_name}**")

# Calculate ₹ Impact
tomorrow_pred = df['Forecast'].iloc[-1]
suggested_stock = int(tomorrow_pred + current_safety_buffer)
capital_required = suggested_stock * item_price

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted Demand", f"{int(tomorrow_pred)} units")
with col2:
    st.metric("AI Suggested Stock", f"{suggested_stock} units", delta=f"+{int(current_safety_buffer)} Safety Buffer")
with col3:
    st.metric("Capital to Invest Today", f"₹ {capital_required:,}", delta_color="off")

# --- 5. INTERACTIVE GRAPH ---
st.markdown("---")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Sales'], mode='lines+markers', name='Actual Sales', line=dict(color='black')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Forecast'], mode='lines', name='Standard Prediction', line=dict(color='blue', dash='dash')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Optimized_Stock'], mode='lines', name='AI Safe Stock Level', line=dict(color='green')))

# The "Loss Prevention" Zone
fig.add_trace(go.Scatter(x=df['Date'].tolist() + df['Date'].tolist()[::-1],
                         y=df['Optimized_Stock'].tolist() + df['Forecast'].tolist()[::-1],
                         fill='toself', fillcolor='rgba(16, 185, 129, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                         name='Profit Protection Zone'))

fig.update_layout(title=f"30-Day Trend & Optimization for {item_name}", template="plotly_white", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.success(f"**AI Business Insight:** By keeping {suggested_stock} units of {item_name} in stock, you protect an estimated **₹ {int(current_safety_buffer * item_price):,}** in revenue against sudden local demand spikes.")

