
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Powering Sustainability",
    layout="wide",
    page_icon="🔋"
)

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

cleaned_data = load_data()
rf, scaler = load_model()

st.sidebar.title("🔌 Navigation")
selection = st.sidebar.radio("Go to", [
    "Project Overview",
    "Visual Explorations",
    "Model Performance",
    "Feature Importance"
])

def show_overview(df):
    st.title("🔋 Powering Sustainability")
    st.markdown("""
    ### Forecasting Electricity Prices and Renewable Energy Trends (2018–2023)

    This dashboard uses machine learning and data visualization to explore electricity pricing and generation trends in Turkey.

    **Goals:**
    - Forecast electricity prices using ML models
    - Understand consumption and seasonal usage patterns
    - Evaluate the impact of renewable energy sources
    - Provide insights for sustainability-focused energy planning

    **Dataset:** Hourly electricity consumption, generation by source, and market prices (TRY, USD, EUR) from 2018 to 2023.
    """)

    st.subheader("📊 Sample of Cleaned Dataset")
    st.dataframe(df.head())

    st.subheader("📌 Dataset Summary")
    st.markdown(f"**Total records:** {df.shape[0]}")
    st.markdown(f"**Total features:** {df.shape[1]}")

    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit")

def show_visuals(df):
    st.title("📊 Visual Explorations")
    st.markdown("Explore electricity demand, generation sources, and pricing patterns interactively.")

    # Monthly Consumption
    st.subheader("🔌 Monthly Electricity Consumption")
    monthly = df.groupby('month')['consumption_MWh'].mean().reset_index()
    fig_monthly = px.bar(monthly, x='month', y='consumption_MWh',
                         labels={'month': 'Month', 'consumption_MWh': 'Average Consumption (MWh)'})
    st.plotly_chart(fig_monthly, use_container_width=True)

    # NEW: Hourly Consumption
    st.subheader("⏰ Hourly Electricity Consumption")
    hourly_consumption = df.groupby('hour')['consumption_MWh'].mean().reset_index()
    fig_hourly = px.line(hourly_consumption, x='hour', y='consumption_MWh',
                         title='Average Hourly Electricity Consumption (MWh)',
                         labels={'consumption_MWh': 'Avg Consumption (MWh)', 'hour': 'Hour of Day'})
    st.plotly_chart(fig_hourly, use_container_width=True)

    # NEW: Weekday vs Weekend
    if 'day_of_week' in df.columns:
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
        weekend_consumption = df.groupby('is_weekend')['consumption_MWh'].mean().reset_index()
        st.subheader("📅 Weekday vs Weekend Electricity Consumption")
        fig_weekend = px.bar(weekend_consumption, x='is_weekend', y='consumption_MWh',
                             title='Weekday vs Weekend Electricity Consumption',
                             labels={'is_weekend': 'Day Type', 'consumption_MWh': 'Avg Consumption (MWh)'})
        st.plotly_chart(fig_weekend, use_container_width=True)

    # NEW: Seasonal Consumption
    if 'season' in df.columns:
        df['season_name'] = df['season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
        seasonal_consumption = df.groupby('season_name')['consumption_MWh'].mean().reset_index()
        st.subheader("🍂 Seasonal Electricity Consumption")
        fig_seasonal = px.bar(seasonal_consumption, x='season_name', y='consumption_MWh',
                              title='Average Seasonal Electricity Consumption',
                              labels={'season_name': 'Season', 'consumption_MWh': 'Avg Consumption (MWh)'})
        st.plotly_chart(fig_seasonal, use_container_width=True)

    # Renewable vs Fossil Share
    st.subheader("⚡ Renewable vs Fossil Generation Share")
    fossil_cols = ['natural_gas', 'coal_imported', 'fuel_oil', 'asphaltite_coal', 'hard_coal']
    renewable_cols = ['solar', 'wind', 'hydro_dam', 'hydro_river', 'biomass', 'geothermal', 'waste_heat']

    fossil_cols = [col for col in fossil_cols if col in df.columns]
    renewable_cols = [col for col in renewable_cols if col in df.columns]

    total_renewable = df[renewable_cols].sum().sum()
    total_fossil = df[fossil_cols].sum().sum()

    fig_pie = px.pie(names=['Fossil', 'Renewable'],
                     values=[total_fossil, total_renewable],
                     title="Electricity Generation Share")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Solar & Wind by Hour
    st.subheader("🌞 Solar and Wind Generation by Hour")
    hourly = df.groupby('hour')[[col for col in ['solar', 'wind'] if col in df.columns]].mean().reset_index()
    fig_hour = px.line(hourly, x='hour', y=hourly.columns[1:],
                       labels={'value': 'Avg Generation (MWh)', 'hour': 'Hour'})
    st.plotly_chart(fig_hour, use_container_width=True)

    # Annual Price Trends
    st.subheader("💸 Annual Electricity Prices (TRY/MWh)")
    annual = df.groupby('year')['TRY/MWh'].mean().reset_index()
    fig_price = px.line(annual, x='year', y='TRY/MWh', markers=True,
                        labels={'TRY/MWh': 'Avg Price (TRY/MWh)'})
    st.plotly_chart(fig_price, use_container_width=True)

def show_model_results(df, rf_model, scaler):
    st.title("🎯 Model Performance")
    st.markdown("Evaluate and compare multiple models on electricity price prediction.")

    X = df.drop(columns=['TRY/MWh'])
    y = df['TRY/MWh']
    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    y_pred_rf = rf_model.predict(X_test_scaled)

    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    model_option = st.selectbox("Choose Model to View:", ["Linear Regression", "Random Forest", "Compare Both"])

    if model_option == "Linear Regression":
        st.subheader("🔹 Linear Regression Results")
        st.metric("R² Score", f"{r2_lr:.3f}")
        st.metric("MAE (TRY)", f"{mae_lr:.2f}")
        st.metric("RMSE (TRY)", f"{rmse_lr:.2f}")

        st.subheader("📈 Actual vs Predicted (Linear Regression)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred_lr, alpha=0.4, edgecolor='k', color='green')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual Price (TRY/MWh)")
        ax.set_ylabel("Predicted Price (TRY/MWh)")
        ax.set_title("Actual vs Predicted Prices – Linear Regression")
        ax.grid(True)
        st.pyplot(fig)

    elif model_option == "Random Forest":
        st.subheader("🔸 Random Forest Results")
        st.metric("R² Score", f"{r2_rf:.3f}")
        st.metric("MAE (TRY)", f"{mae_rf:.2f}")
        st.metric("RMSE (TRY)", f"{rmse_rf:.2f}")

        st.subheader("📈 Actual vs Predicted (Random Forest)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred_rf, alpha=0.4, edgecolor='k', color='royalblue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual Price (TRY/MWh)")
        ax.set_ylabel("Predicted Price (TRY/MWh)")
        ax.set_title("Actual vs Predicted Prices – Random Forest")
        ax.grid(True)
        st.pyplot(fig)

    else:
        st.subheader("🔍 Comparison of Models")
        comparison_df = pd.DataFrame({
            "Model": ["Random Forest", "Linear Regression"],
            "R²": [r2_rf, r2_lr],
            "MAE": [mae_rf, mae_lr],
            "RMSE": [rmse_rf, rmse_lr]
        })
        st.dataframe(comparison_df)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred_rf, alpha=0.4, edgecolor='k', label='Random Forest', color='royalblue')
        ax.scatter(y_test, y_pred_lr, alpha=0.4, edgecolor='gray', label='Linear Regression', color='green')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
        ax.set_xlabel("Actual Price (TRY/MWh)")
        ax.set_ylabel("Predicted Price (TRY/MWh)")
        ax.set_title("Actual vs Predicted – Model Comparison")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

def show_feature_importance(df, model):
    st.title("🧠 Feature Importance")
    st.markdown("Understanding which features most influence electricity price predictions.")

    X = df.drop(columns=['TRY/MWh'])
    X = X.select_dtypes(include=[np.number])
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = np.array(X.columns)[indices]
    sorted_importances = importances[indices]

    # Interactive Top-N slider
    st.subheader("🎯 Select Top N Features")
    top_n = st.slider("Select number of top features to display:", min_value=5, max_value=len(sorted_features), value=15)

    # Top-N Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_features[:top_n][::-1], sorted_importances[:top_n][::-1], color='skyblue')
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Most Important Features")
    st.pyplot(fig)

    # Cumulative Importance Plot
    st.subheader("📈 Cumulative Feature Importance")
    cumulative_importance = np.cumsum(sorted_importances)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(cumulative_importance)+1), cumulative_importance, marker='o')
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Cumulative Importance")
    ax.set_title("Cumulative Feature Importance")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Correlation Heatmap of Top-N
    st.subheader("🧱 Correlation Heatmap of Top Features")
    selected_top = sorted_features[:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[selected_top].corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap of Top Features")
    st.pyplot(fig)

    # Show raw data (optional)
    if st.checkbox("📋 Show full importance table"):
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        st.dataframe(importance_df.reset_index(drop=True))

# Render page
if selection == "Project Overview":
    show_overview(cleaned_data)
elif selection == "Visual Explorations":
    show_visuals(cleaned_data)
elif selection == "Model Performance":
    show_model_results(cleaned_data, rf, scaler)
elif selection == "Feature Importance":
    show_feature_importance(cleaned_data, rf)
