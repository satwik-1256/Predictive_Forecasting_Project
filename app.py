import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA

# PAGE CONFIG
st.set_page_config(
    page_title="Predictive Forecasting Dashboard",
    layout="wide")

# TITLE
st.title("Predictive Forecasting of HHS Care Load")
st.write(
    """
    Interactive forecasting dashboard for analyzing
    HHS Unaccompanied Children Program data.
    """)

# LOAD DATA
df = pd.read_csv(
    "HHS_Unaccompanied_Alien_Children_Program.csv")

# DATA CLEANING
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
numeric_columns = [
    'Children apprehended and placed in CBP custody*',
    'Children in CBP custody',
    'Children transferred out of CBP custody',
    'Children in HHS Care',
    'Children discharged from HHS Care']
for col in numeric_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '', regex=False)
        .astype(float))
df = df.dropna()

# FEATURE ENGINEERING
df['lag_1'] = (
    df['Children in HHS Care']
    .shift(1))
df['lag_7'] = (
    df['Children in HHS Care']
    .shift(7))
df['rolling_mean_7'] = (
    df['Children in HHS Care']
    .rolling(window=7)
    .mean())
df = df.dropna()

# FORECAST HORIZON SELECTOR
forecast_days = st.sidebar.selectbox(
    "Select Forecast Horizon",
    [7, 14, 30])

# MODEL TOGGLE
selected_model = st.sidebar.selectbox(
    "Select Forecasting Model",
    [
        "Random Forest",
        "Gradient Boosting",
        "Moving Average",
        "ARIMA"])

# TRAIN TEST SPLIT
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]
features = [
    'lag_1',
    'lag_7',
    'rolling_mean_7']
X_train = train[features]
y_train = train['Children in HHS Care']
X_test = test[features]
y_test = test['Children in HHS Care']

# RANDOM FOREST MODEL
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# GRADIENT BOOSTING MODEL
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

# ARIMA MODEL
arima_train = train['Children in HHS Care']
arima_model = ARIMA(
    arima_train,
    order=(2,1,2))
arima_result = arima_model.fit()
arima_predictions = arima_result.forecast(
    steps=len(test))

# MOVING AVERAGE MODEL
moving_avg_predictions = (
    test['rolling_mean_7'])

# MODEL SELECTION
if selected_model == "Random Forest":
    predictions = rf_predictions
elif selected_model == "Gradient Boosting":
    predictions = gb_predictions
elif selected_model == "ARIMA":
    predictions = arima_predictions
else:
    predictions = moving_avg_predictions

# EVALUATION METRICS
mae = mean_absolute_error(
    y_test,
    predictions)
rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions))
accuracy = 100 - (
    np.mean(
        np.abs(
            (y_test - predictions) / y_test)) * 100)

# KPI SECTION
st.subheader("KPI Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Forecast Accuracy",
    f"{accuracy:.2f}%")
col2.metric(
    "MAE",
    f"{mae:.2f}")
col3.metric(
    "RMSE",
    f"{rmse:.2f}")
col4.metric(
    "Average Care Load",
    f"{df['Children in HHS Care'].mean():.0f}")

# FUTURE CARE LOAD FORECAST
st.subheader("Future Care Load Forecast")
fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.plot(
    test['Date'],
    y_test,
    label='Actual')
ax1.plot(
    test['Date'],
    predictions,
    label='Predicted')
ax1.set_title(
    f"{selected_model} Forecast")
ax1.legend()
st.pyplot(fig1)

# CONFIDENCE INTERVAL VISUALIZATION
st.subheader("Confidence Interval Visualization")
upper_bound = predictions + rmse
lower_bound = predictions - rmse
fig2, ax2 = plt.subplots(figsize=(14,6))
ax2.plot(
    test['Date'],
    predictions,
    label='Predicted')
ax2.fill_between(
    test['Date'],
    lower_bound,
    upper_bound,
    alpha=0.3,
    label='Confidence Interval')
ax2.legend()
st.pyplot(fig2)

# DISCHARGE DEMAND FORECAST PANEL
st.subheader("Discharge Demand Forecast")
fig3, ax3 = plt.subplots(figsize=(14,6))
ax3.plot(
    df['Date'],
    df['Children discharged from HHS Care'],
    color='green')
ax3.set_title(
    "Children Discharged from HHS Care")
st.pyplot(fig3)

# MONTHLY TREND ANALYSIS
st.subheader("Monthly Trend Analysis")
df['Month'] = df['Date'].dt.month
monthly_avg = df.groupby(
    'Month'
)['Children in HHS Care'].mean()
fig4, ax4 = plt.subplots(figsize=(10,5))
monthly_avg.plot(
    kind='bar',
    ax=ax4)
ax4.set_title(
    "Monthly Average Care Load")
st.pyplot(fig4)

# SCENARIO COMPARISON TABLE
st.subheader("Scenario Comparison")
comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Random Forest': rf_predictions,
    'Gradient Boosting': gb_predictions,
    'Moving Average': moving_avg_predictions.values,
    'ARIMA': arima_predictions.values})
st.dataframe(comparison_df.head(20))

# MODEL COMPARISON METRICS
st.subheader("Model Comparison Metrics")
rf_mae = mean_absolute_error(
    y_test,
    rf_predictions)
gb_mae = mean_absolute_error(
    y_test,
    gb_predictions)
ma_mae = mean_absolute_error(
    y_test,
    moving_avg_predictions)
arima_mae = mean_absolute_error(
    y_test,
    arima_predictions)
rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions))
gb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gb_predictions))
ma_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        moving_avg_predictions))
arima_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        arima_predictions))
metrics_df = pd.DataFrame({
    'Model': [
        'Random Forest',
        'Gradient Boosting',
        'Moving Average',
        'ARIMA'],
    'MAE': [
        round(rf_mae, 2),
        round(gb_mae, 2),
        round(ma_mae, 2),
        round(arima_mae, 2)],
    'RMSE': [
        round(rf_rmse, 2),
        round(gb_rmse, 2),
        round(ma_rmse, 2),
        round(arima_rmse, 2)]})
st.table(metrics_df)

# SCENARIO COMPARISON VISUALIZATION
st.subheader("Scenario Comparison Visualization")
fig5, ax5 = plt.subplots(figsize=(14,6))

# ACTUAL VALUES
ax5.plot(
    test['Date'],
    y_test,
    label='Actual',
    linewidth=2,
    color='black')

# RANDOM FOREST
ax5.plot(
    test['Date'],
    rf_predictions,
    label='Random Forest',
    linestyle='--',
    linewidth=3,
    color='red')

# GRADIENT BOOSTING
ax5.plot(
    test['Date'],
    gb_predictions,
    label='Gradient Boosting',
    linestyle='-.',
    linewidth=2,
    color='blue')

# MOVING AVERAGE
ax5.plot(
    test['Date'],
    moving_avg_predictions,
    label='Moving Average',
    linestyle=':',
    linewidth=2,
    color='green')

# ARIMA
ax5.plot(
    test['Date'],
    arima_predictions,
    label='ARIMA',
    linestyle='solid',
    linewidth=2,
    color='orange')
ax5.set_title(
    "Forecasting Model Scenario Comparison")
ax5.set_xlabel("Date")
ax5.set_ylabel("Care Load")
ax5.legend()
ax5.grid(True)
st.pyplot(fig5)

# DATASET PREVIEW
st.subheader("Dataset Preview")
st.dataframe(df.head(20))

# DOWNLOAD CLEANED DATASET
csv = df.to_csv(index=False)
st.download_button(
    label="Download Cleaned Dataset",
    data=csv,
    file_name='cleaned_forecasting_data.csv',
    mime='text/csv')