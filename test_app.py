import streamlit as st
import pandas as pd
import numpy as np

st.title("Smart Store AI - Dashboard")

# --- THE TOGGLE TRICK ---
# Change this to False when you are ready for the real world!
USE_FAKE_DATA = True 

@st.cache_data # This keeps the app fast by not regenerating data on every click
def load_data(is_fake):
    if is_fake:
        # 1. Generate 100 rows of fake store data for testing
        return pd.DataFrame({
            "Product": np.random.choice(["Smartphone", "Headphones", "Laptop", "Smartwatch"], 100),
            "Price (USD)": np.random.randint(50, 1200, 100),
            "Units Sold": np.random.randint(1, 10, 100),
            "Customer Satisfaction": np.random.uniform(3.0, 5.0, 100).round(1)
        })
    else:
        # 2. Your Real-World Code goes here later!
        # Example: return pd.read_csv("my_real_store_data.csv") 
        # Or connect to your actual cloud database/AI API
        return pd.DataFrame() 

# --- MAIN APP LOGIC ---
# Load the data based on your toggle
store_data = load_data(USE_FAKE_DATA)

# Display a metric and the table
st.subheader("Store Analytics")
st.metric("Total Items Sold", store_data["Units Sold"].sum())

st.write("Recent Transactions:")
st.dataframe(store_data)

st.markdown("---")
if USE_FAKE_DATA:
    st.info("🟡 Currently running in TEST mode with fake data.")
else:
    st.success("🟢 Currently running in PRODUCTION with real data.")
  
