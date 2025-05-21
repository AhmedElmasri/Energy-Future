import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Powering Sustainability", layout="wide", page_icon="🔋")

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

@st.cache_resource
def load_model():
    if not os.path.exists("random_forest_model.pkl") or not os.path.exists("scaler.pkl"):
        df = load_data()
        X = df.drop(columns=['TRY/MWh']).select_dtypes(include='number')
        y = df['TRY/MWh']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestRegressor(
            n_estimators=25,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        joblib.dump(model, "random_forest_model.pkl", compress=("zlib", 9))
        joblib.dump(scaler, "scaler.pkl")
        print("✅ Trained and saved small model with compression.")

    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

cleaned_data = load_data()
rf_model, scaler = load_model()

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
    st.dataframe(df.head())
    st.markdown(f"**Total records:** {df.shape[0]}")
    st.markdown(f"**Total features:** {df.shape[1]}")
    st.download_button("📥 Download Dataset", df.to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")

def show_visuals(df):
    st.title("📊 Visual Explorations")

    monthly = df.groupby('month')['consumption_MWh'].mean().reset_index()
    st.subheader("🔌 Monthly Electricity Consumption")
    st.plotly_chart(px.bar(monthly, x='month', y='consumption_MWh'), use_container_width=True)

    hourly = df.groupby('hour')['consumption_MWh'].mean().reset_index()
    st.subheader("⏰ Hourly Electricity Consumption")
    st.plotly_chart(px.line(hourly, x='hour', y='consumption_MWh'), use_container_width=True)

    if 'day_of_week' in df.columns:
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
        weekend = df.groupby('is_weekend')['consumption_MWh'].mean().reset_index()
        st.subheader("📅 Weekday vs Weekend")
        st.plotly_chart(px.bar(weekend, x='is_weekend', y='consumption_MWh'), use_container_width=True)

    if 'season' in df.columns:
        df['season_name'] = df['season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
        seasonal = df.groupby('season_name')['consumption_MWh'].mean().reset_index()
        st.subheader("🍂 Seasonal Electricity Consumption")
        st.plotly_chart(px.bar(seasonal, x='season_name', y='consumption_MWh'), use_container_width=True)

    fossil = ['natural_gas', 'coal_imported', 'fuel_oil', 'asphaltite_coal', 'hard_coal']
    renewable = ['solar', 'wind', 'hydro_dam', 'hydro_river', 'biomass', 'geothermal', 'waste_heat']
    fossil = [c for c in fossil if c in df.columns]
    renewable = [c for c in renewable if c in df.columns]

    st.subheader("⚡ Fossil vs Renewable Generation")
    st.plotly_chart(px.pie(names=['Fossil', 'Renewable'], values=[df[fossil].sum().sum(), df[renewable].sum().sum()]), use_container_width=True)

    hour_gen = df.groupby('hour')[[c for c in ['solar', 'wind'] if c in df.columns]].mean().reset_index()
    st.subheader("🌞 Solar and Wind by Hour")
    st.plotly_chart(px.line(hour_gen, x='hour', y=hour_gen.columns[1:]), use_container_width=True)

    annual = df.groupby('year')['TRY/MWh'].mean().reset_index()
    st.subheader("💸 Annual Prices (TRY/MWh)")
    st.plotly_chart(px.line(annual, x='year', y='TRY/MWh', markers=True), use_container_width=True)

def show_model_results(df, model, scaler):
    st.title("🎯 Model Performance")
    X = df.drop(columns=['TRY/MWh']).select_dtypes(include='number')
    y = df['TRY/MWh']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_pred = model.predict(X_test_scaled)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)

    r2_rf = r2_score(y_test, rf_pred)
    mae_rf = mean_absolute_error(y_test, rf_pred)
    rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))

    r2_lr = r2_score(y_test, lr_pred)
    mae_lr = mean_absolute_error(y_test, lr_pred)
    rmse_lr = np.sqrt(mean_squared_error(y_test, lr_pred))

    model_option = st.selectbox("Choose Model to View:", ["Random Forest", "Linear Regression", "Compare Both"])

    if model_option == "Random Forest":
        st.metric("R²", f"{r2_rf:.3f}")
        st.metric("MAE", f"{mae_rf:.2f}")
        st.metric("RMSE", f"{rmse_rf:.2f}")
    elif model_option == "Linear Regression":
        st.metric("R²", f"{r2_lr:.3f}")
        st.metric("MAE", f"{mae_lr:.2f}")
        st.metric("RMSE", f"{rmse_lr:.2f}")
    else:
        st.dataframe(pd.DataFrame({
            "Model": ["Random Forest", "Linear Regression"],
            "R²": [r2_rf, r2_lr],
            "MAE": [mae_rf, mae_lr],
            "RMSE": [rmse_rf, rmse_lr]
        }))

def show_feature_importance(df, model):
    st.title("🧠 Feature Importance")
    X = df.drop(columns=['TRY/MWh']).select_dtypes(include='number')
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = np.array(X.columns)[sorted_idx]
    sorted_importances = importances[sorted_idx]

    top_n = st.slider("Top N features:", 5, len(sorted_features), 15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_features[:top_n][::-1], sorted_importances[:top_n][::-1], color='skyblue')
    ax.set_title("Top Feature Importances")
    st.pyplot(fig)

    st.subheader("📈 Cumulative Importance")
    cumulative = np.cumsum(sorted_importances)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumulative)+1), cumulative, marker='o')
    ax.axhline(0.95, color='r', linestyle='--')
    st.pyplot(fig)

    st.subheader("🧱 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[sorted_features[:top_n]].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    if st.checkbox("📋 Show full importance table"):
        st.dataframe(pd.DataFrame({
            'Feature': sorted_features,
            'Importance': sorted_importances
        }).reset_index(drop=True))

# Page router
if selection == "Project Overview":
    show_overview(cleaned_data)
elif selection == "Visual Explorations":
    show_visuals(cleaned_data)
elif selection == "Model Performance":
    show_model_results(cleaned_data, rf_model, scaler)
elif selection == "Feature Importance":
    show_feature_importance(cleaned_data, rf_model)