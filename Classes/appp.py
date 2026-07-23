
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ CONFIGURATION & DATA LOADING
# ==========================================
st.set_page_config(page_title="Walmart Demand Forecaster", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('walmart_forecaster.pkl')
    data = pd.read_csv('walmart_seed_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return model, data

model, df = load_assets()
features = [
    'Store', 'Dept', 'IsHoliday', 'Weekly_Sales_Lag_1', 'Weekly_Sales_Lag_4', 
    'Weekly_Sales_Lag_52', 'Rolling_Mean_4', 'Rolling_Mean_12', 'Rolling_Std_4',
    'Month', 'Week_of_Year', 'IsHoliday_NextWeek', 'IsHoliday_LastWeek'
]

# ==========================================
# 🎛️ UI SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Forecasting Parameters")
store_id = st.sidebar.selectbox("Select Store", sorted(df['Store'].unique()))
dept_id = st.sidebar.selectbox("Select Department", sorted(df[df['Store'] == store_id]['Dept'].unique()))
horizon = st.sidebar.slider("Forecast Horizon (Weeks)", min_value=1, max_value=12, value=4)

st.title("📈 Retail Demand Forecasting Engine")
st.markdown("Autoregressive recursive forecasting using LightGBM.")

# ==========================================
# 🔮 RECURSIVE FORECASTING ENGINE
# ==========================================
if st.button("Generate Out-of-Sample Forecast"):
    with st.spinner('Computing temporal lags and projecting vectors...'):
        
        # 1. Isolate Seed Data
        seed_data = df[(df['Store'] == store_id) & (df['Dept'] == dept_id)].sort_values('Date')
        
        if seed_data.empty:
            st.error("No historical data available for this Store/Dept combination.")
        else:
            last_date = seed_data['Date'].max()
            
            # 2. Build Future Horizon Space
            future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon, freq='W-FRI')
            future_df = pd.DataFrame({
                'Date': future_dates, 'Store': store_id, 'Dept': dept_id,
                'Weekly_Sales': np.nan, 'IsHoliday': 0
            })
            
            sim_horizon = pd.concat([seed_data, future_df], ignore_index=True).sort_values('Date').reset_index(drop=True)
            
            # 3. Recursive Loop
            for i in range(len(sim_horizon)):
                curr_date = sim_horizon.loc[i, 'Date']
                if curr_date > last_date:
                    sim_horizon.loc[i, 'Weekly_Sales_Lag_1'] = sim_horizon.loc[i - 1, 'Weekly_Sales']
                    sim_horizon.loc[i, 'Weekly_Sales_Lag_4'] = sim_horizon.loc[i - 4, 'Weekly_Sales']
                    sim_horizon.loc[i, 'Weekly_Sales_Lag_52'] = sim_horizon.loc[i - 52, 'Weekly_Sales']
                    
                    past_4 = sim_horizon.loc[i-4:i-1, 'Weekly_Sales']
                    sim_horizon.loc[i, 'Rolling_Mean_4'] = past_4.mean()
                    sim_horizon.loc[i, 'Rolling_Std_4'] = past_4.std()
                    sim_horizon.loc[i, 'Rolling_Mean_12'] = sim_horizon.loc[i-12:i-1, 'Weekly_Sales'].mean()
                    
                    sim_horizon.loc[i, 'Month'] = curr_date.month
                    sim_horizon.loc[i, 'Week_of_Year'] = int(curr_date.isocalendar()[1])
                    sim_horizon.loc[i, 'IsHoliday_NextWeek'] = 0
                    sim_horizon.loc[i, 'IsHoliday_LastWeek'] = sim_horizon.loc[i - 1, 'IsHoliday']
                    
                    X_future = sim_horizon.loc[[i], features]
                    pred = model.predict(X_future)[0]
                    sim_horizon.loc[i, 'Weekly_Sales'] = max(0, pred)

            # ==========================================
            # 📊 VISUALIZATION LAYER
            # ==========================================
            st.success("Forecast Generation Complete!")
            
            hist_plot = sim_horizon[sim_horizon['Date'] <= last_date].tail(52)
            fut_plot = sim_horizon[sim_horizon['Date'] > last_date]
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(hist_plot['Date'], hist_plot['Weekly_Sales'], label='Historical Sales', color='#1f77b4', linewidth=2)
            ax.plot(fut_plot['Date'], fut_plot['Weekly_Sales'], label='Out-of-Sample Forecast', color='#ff7f0e', linewidth=2, linestyle='--')
            
            # Connect the two lines visually
            connector_x = [hist_plot['Date'].iloc[-1], fut_plot['Date'].iloc[0]]
            connector_y = [hist_plot['Weekly_Sales'].iloc[-1], fut_plot['Weekly_Sales'].iloc[0]]
            ax.plot(connector_x, connector_y, color='#ff7f0e', linewidth=2, linestyle='--')

            ax.set_title(f"Demand Trajectory: Store {store_id} | Dept {dept_id}", fontweight='bold')
            ax.set_ylabel("Weekly Sales ($)")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            
            st.pyplot(fig)
            
            # Show Raw Data Table
            st.subheader("Raw Forecast Vector")
            st.dataframe(fut_plot[['Date', 'Weekly_Sales']].style.format({'Weekly_Sales': '${:,.2f}'}))