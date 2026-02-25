import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta

# --- 1. UI SETTINGS ---
st.set_page_config(page_title="Pro Store AI", layout="wide", initial_sidebar_state="expanded")
st.title("₹ Advanced Profit & Horizon Optimizer")

DATA_FILE = "live_sales_data.csv"

# --- 2. DATA STORAGE & RECORDING ---
def load_data():
    if not os.path.exists(DATA_FILE):
        dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=40)
        # Creating a realistic baseline with some momentum
        sales = np.random.randint(10, 25, size=40) + np.sin(np.arange(40)) * 5
        df = pd.DataFrame({'Date': dates, 'Sales': np.maximum(0, sales.astype(int))})
        df.to_csv(DATA_FILE, index=False)
    
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# --- 3. INTERACTIVE SIDEBAR ---
with st.sidebar:
    st.header("🏪 Store Settings")
    item_name = st.text_input("Target Item", "Smartphones")
    item_price = st.number_input("Avg Price (₹)", value=15000, step=500)
    risk_tolerance =   st.slider("Select Risk Tolerance", min_value=1, max_value=10)
    st.markdown("---")
    st.header("📂 Train on Real Data")
    st.caption("Upload historical sales (CSV) to retrain the AI instantly.")
    uploaded_file = st.file_uploader("Upload Shop's CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            real_df = pd.read_csv(uploaded_file)
            # Ensure the CSV has 'Date' and 'Sales' columns
            if 'Date' in real_df.columns and 'Sales' in real_df.columns:
                real_df['Date'] = pd.to_datetime(real_df['Date'])
                # Overwrite the dummy database with real data
                real_df.to_csv(DATA_FILE, index=False)
                st.success("✅ Real data loaded! AI retraining...")
                st.rerun()
            else:
                st.error("CSV must have 'Date' and 'Sales' columns.")
        except Exception as e:
            st.error("Error reading file.")
            

# --- 4. ADVANCED AI CORE (Gradient Boosting + Lags) ---
def engineer_features(data):
    # Creating "Momentum" variables for the AI
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Lag_1'] = data['Sales'].shift(1) # Yesterday's sales
    data['Lag_2'] = data['Sales'].shift(2) # Day before yesterday
    data['Rolling_3'] = data['Sales'].rolling(window=3).mean()
    return data.dropna()

def run_horizon_prediction(data, h_days=6):
    df_feat = engineer_features(data.copy())
    
    if len(df_feat) < 10:
        return None, None, "Not enough data"

    # Train the Gradient Boosting Engine
    X = df_feat[['DayOfWeek', 'Lag_1', 'Lag_2', 'Rolling_3']]
    y = df_feat['Sales']
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Calculate historical Residuals (Errors)
    df_feat['Forecast'] = model.predict(X)
    df_feat['Residual'] = df_feat['Sales'] - df_feat['Forecast']
    sigma_residual = df_feat['Residual'].std()

    # --- FORECAST h DAYS INTO THE FUTURE ---
    future_dates = [data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, h_days + 1)]
    future_preds = []
    
    # We use the last known values to start our rolling prediction
    last_sales = data['Sales'].values[-3:]
    
    for i in range(h_days):
        next_day_of_week = future_dates[i].dayofweek
        lag_1 = last_sales[-1]
        lag_2 = last_sales[-2]
        roll_3 = np.mean(last_sales[-3:])
        
        pred = model.predict([[next_day_of_week, lag_1, lag_2, roll_3]])[0]
        future_preds.append(max(0, pred))
        # Update our 'last known' list with the new prediction to feed the next loop
        last_sales = np.append(last_sales, pred)

    # Formal Safety Stock Math for horizon h: SS = Z * sigma * sqrt(h)
    Z_score = (11 - risk_tolerance) / 3.0 # Maps risk tolerance to a multiplier
    safety_stock_total = Z_score * sigma_residual * np.sqrt(h_days)
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Base': future_preds,
        'Upstock_Target': np.array(future_preds) + (safety_stock_total / h_days) # Spreading safety buffer
    })
    
    return df_feat, future_df, "Success"

df_historical, df_future, status = run_horizon_prediction(df)
# --- 5. DASHBOARD & METRICS ---
if status == "Success":
    st.markdown("### 6-Day Horizon Optimization")
    
    total_6_day_demand = int(df_future['Predicted_Base'].sum())
    total_suggested_order = int(df_future['Upstock_Target'].sum())
    capital_req = total_suggested_order * item_price

    # --- THE "SHOPKEEPER PSYCHOLOGY" MATH ---
    # A normal shopkeeper orders based on their recent "best" day to be safe.
    naive_daily_guess = df_historical['Sales'].tail(7).max() 
    naive_6_day_order = int(naive_daily_guess * 6)
    
    # Calculate how many dead-stock units the AI just saved them from buying
    units_saved = max(0, naive_6_day_order - total_suggested_order)
    cash_saved = int(units_saved * item_price)

    # --- THE DISPLAY ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted 6-Day Demand", f"{total_6_day_demand} units")
    col2.metric("AI Recommended Order", f"{total_suggested_order} units", delta=f"+{total_suggested_order - total_6_day_demand} Math Buffer")
    col3.metric("Required Capital", f"₹ {capital_req:,}")

    # THE BIG REVEAL FOR THE OWNER
    if cash_saved > 0:
        st.success(f"🛡️ **Cash Protected:** By following the AI instead of your gut feeling, you avoided buying {units_saved} excess units. You just kept **₹ {cash_saved:,}** safely in your bank account instead of locked in dead stock.")
    else:
        st.info("📈 **Growth Mode:** Demand is spiking! Your capital is being perfectly allocated to prevent empty shelves and lost customers.")

    # --- 6. ADVANCED GRAPHICAL PLOT ---
    st.markdown("---")
    fig = go.Figure()
    
    # Historical Data
    fig.add_trace(go.Scatter(x=df_historical['Date'].tail(15), y=df_historical['Sales'].tail(15), mode='lines+markers', name='Actual Sales', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df_historical['Date'].tail(15), y=df_historical['Forecast'].tail(15), mode='lines', name='AI Fit', line=dict(color='blue', dash='dash')))
    
    # Future 6-Day Forecast
    fig.add_trace(go.Scatter(x=df_future['Date'], y=df_future['Predicted_Base'], mode='lines+markers', name='6-Day Base Forecast', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_future['Date'], y=df_future['Upstock_Target'], mode='lines', name='Optimized Upstock', line=dict(color='green', width=3)))

    # Future Safety Zone (The mathematical buffer)
    fig.add_trace(go.Scatter(
        x=df_future['Date'].tolist() + df_future['Date'].tolist()[::-1],
        y=df_future['Upstock_Target'].tolist() + df_future['Predicted_Base'].tolist()[::-1],
        fill='toself', fillcolor='rgba(16, 185, 129, 0.3)', line=dict(color='rgba(255,255,255,0)'),
        name='Capital Safety Zone'
    ))

    fig.update_layout(title="Multi-Day Horizon Forecast with Formal Safety Stock", template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please record at least 10 days of data to activate the Advanced Horizon Engine.")

    
