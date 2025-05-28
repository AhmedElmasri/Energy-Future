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
        X = df[['hour', 'day_of_week', 'month', 'season', 'consumption_MWh', 'natural_gas', 'lignite', 'coal_imported', 'fuel_oil', 'wind', 'solar', 'hydro_dam', 'hydro_river', 'biomass', 'geothermal', 'total_renewable_enegry_MWh', 'total_nonrenewable_enegry_MWh', 'renewable_share', 'TRY/MWh']]
        y = df['renewable_strength_score']
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
    st.title("🔋 Energy Future")
    st.markdown("""
    ### Predicting the Clean Energy Future: Strategic Modeling for Renewable Transformation

    **Objective:**
    
    This proposal presents a comprehensive, data-driven strategy to advocate for renewable energy as the foundation of future energy systems. By combining historical data analysis (2018–2023) with advanced machine learning forecasting, the evidence reveals that renewable energy is not only clean and emission-free, but also increasingly reliable, cost-efficient, and capable of delivering sustainable energy supply year-round. The centerpiece of this analysis is the creation and modeling of a unified target variable: renewable_strength_score, which quantifies renewable performance in operational and economic terms.

    """)
    st.dataframe(df.head())
    st.markdown(f"**Total records:** {df.shape[0]}")
    st.markdown(f"**Total features:** {df.shape[1]}")
    st.download_button("📥 Download Dataset", df.to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")

def show_visuals(df):
    st.title("📊 Visual Explorations")

    total_renewable_enegry_per_year_MWh = df.groupby('year')['total_renewable_enegry_MWh'].sum().reset_index()
    total_renewable_enegry_per_year_MWh['total_renewable_enegry_MWh'] = total_renewable_enegry_per_year_MWh['total_renewable_enegry_MWh'].apply('{:.2f}'.format)
    st.subheader("🔌 Yearly Renewable Electricity Generation")
    st.plotly_chart(px.bar(total_renewable_enegry_per_year_MWh, x='total_renewable_enegry_MWh', y='year', title='Total Renewable Energy per Year (MWh)', labels={'total_renewable_enegry_MWh': 'Renewable Energy (MWh)', 'year': 'Year'}, orientation='h'), use_container_width=True)

    total_nonrenewable_enegry_per_year_MWh = df.groupby('year')['total_nonrenewable_enegry_MWh'].sum().reset_index()
    total_nonrenewable_enegry_per_year_MWh['total_nonrenewable_enegry_MWh'] = total_nonrenewable_enegry_per_year_MWh['total_nonrenewable_enegry_MWh'].apply('{:.2f}'.format)
    st.subheader("🔌 Yearly Nonrenewable Electricity Generation")
    st.plotly_chart(px.bar(total_nonrenewable_enegry_per_year_MWh, x='total_nonrenewable_enegry_MWh', y='year'), use_container_width=True)

    renewable_cols = ['hydro_dam', 'hydro_river', 'solar', 'wind', 'biomass', 'geothermal', 'waste_heat', 'international']
    total_renewable_each_column_per_year_MWh = df.groupby('year')[renewable_cols].sum().reset_index()
    st.subheader("🔌 Electricity Generation for Each Renewable Resource")
    st.plotly_chart(px.line(total_renewable_each_column_per_year_MWh, x='year', y=renewable_cols, title='Yearly Contribution of Each Renewable Source', labels={'value': 'MWh', 'variable': 'Energy Source'}), use_container_width=True)

    nonrenewable_cols = ['natural_gas', 'lignite', 'asphaltite_coal', 'hard_coal', 'fuel_oil', 'coal_imported']
    total_nonrenewable_each_column_per_year_MWh = df.groupby('year')[nonrenewable_cols].sum().reset_index()
    melted_nonrenew = total_nonrenewable_each_column_per_year_MWh.melt(id_vars='year', value_vars=nonrenewable_cols, var_name='Source', value_name='MWh')
    st.subheader("🔌 Electricity Generation for Each Nonrenewable Resource")
    st.plotly_chart(px.line(melted_nonrenew, x='year', y='MWh', color='Source', title='Yearly Contribution of Each Nonrenewable Source', labels={'value': 'MWh', 'variable': 'Energy Source'}), use_container_width=True)

    renewable_cols = ['hydro_dam', 'hydro_river', 'solar', 'wind', 'biomass', 'geothermal', 'waste_heat', 'international']
    total_renewable_each_column_per_season_MWh = df.groupby('season')[renewable_cols].sum().reset_index()
    season_labels = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    total_renewable_each_column_per_season_MWh['season'] = total_renewable_each_column_per_season_MWh['season'].map(season_labels)
    melted_seasonal = total_renewable_each_column_per_season_MWh.melt(id_vars='season', value_vars=renewable_cols, var_name='Source', value_name='MWh')
    st.subheader("🔌 Seasonal Renewable Electricity Generation")
    st.plotly_chart(px.bar(melted_seasonal, x='season', y='MWh', color='Source', title='Seasonal Renewable Energy by Type', barmode='group', labels={'MWh': 'MWh', 'season': 'Season', 'Source': 'Energy Source'}), use_container_width=True)

    total_nonrenewable_each_column_per_season_MWh = df.groupby('season')[nonrenewable_cols].sum().reset_index()
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    total_nonrenewable_each_column_per_season_MWh['season'] = total_nonrenewable_each_column_per_season_MWh['season'].map(season_labels)
    melted_seasonal_nonrenew = total_nonrenewable_each_column_per_season_MWh.melt(id_vars='season', value_vars=nonrenewable_cols, var_name='Source', value_name='MWh')
    st.subheader("🔌 Seasonal Nonrenewable Electricity Generation")
    st.plotly_chart(px.bar(melted_seasonal_nonrenew, x='season', y='MWh', color='Source', title='Seasonal Nonrenewable Energy by Type', barmode='group', labels={'value': 'MWh', 'season': 'Season'}), use_container_width=True)

    season_avg_MWh = df.groupby('season')['consumption_MWh'].mean().reset_index()
    season_avg_MWh['season'] = season_avg_MWh['season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
    st.subheader("⚡ Electricity Consumption through Seasons")
    st.plotly_chart(px.pie(season_avg_MWh, names='season', values='consumption_MWh', title='Average Electricity Demand by Season'), use_container_width=True)

    yearly_average_energy_consumption = df.groupby('year')['consumption_MWh'].mean().reset_index()
    highest_average_energy_consumption = yearly_average_energy_consumption.loc[yearly_average_energy_consumption['consumption_MWh'].idxmax()]
    st.subheader("📈 Highest Electricity Consumption")
    st.plotly_chart(px.bar(yearly_average_energy_consumption, x='year', y='consumption_MWh', title='Average Electricity Consumption by Year', labels={'consumption_MWh': 'Avg Consumption (MWh)'}), use_container_width=True)

    renewables = ['hydro_dam', 'hydro_river', 'solar', 'wind', 'biomass', 'geothermal', 'waste_heat', 'international']
    renewable_by_year = df[df['year'].isin([2018, 2023])].groupby('year')[renewables].sum().T
    renewable_by_year.columns = ['2018', '2023']
    renewable_by_year['Growth (MWh)'] = renewable_by_year['2023'] - renewable_by_year['2018']
    renewable_by_year = renewable_by_year.sort_values('Growth (MWh)', ascending=False).reset_index().rename(columns={'index': 'Source'})
    st.subheader("📈 The Most Renewable Source Growth")
    st.plotly_chart(px.bar(renewable_by_year, x='Source', y='Growth (MWh)', title='Renewable Energy Growth (2018–2023)', color='Growth (MWh)', color_continuous_scale='Greens'), use_container_width=True)

    nonrenewables = ['natural_gas', 'lignite', 'asphaltite_coal', 'hard_coal', 'fuel_oil', 'coal_imported']
    nonrenew_by_year = df[df['year'].isin([2018, 2023])].groupby('year')[nonrenewables].sum().T
    nonrenew_by_year.columns = ['2018', '2023']
    nonrenew_by_year['Growth (MWh)'] = nonrenew_by_year['2023'] - nonrenew_by_year['2018']
    nonrenew_by_year = nonrenew_by_year.sort_values('Growth (MWh)', ascending=False).reset_index().rename(columns={'index': 'Source'})
    st.subheader("📈 The Most Nonrenewable Source Growth")
    st.plotly_chart(px.bar(nonrenew_by_year, x='Source', y='Growth (MWh)', title='Nonrenewable Energy Growth (2018–2023)', color='Growth (MWh)', color_continuous_scale='Reds'), use_container_width=True)

    solar_wind_variability_year = df.groupby('year')[['solar', 'wind']].std().reset_index()
    solar_wind_variability_hour = df.groupby('hour')[['solar', 'wind']].std().reset_index()
    st.subheader("📈 Variability of Solar and Wind Through Day and Year")
    st.plotly_chart(px.line(solar_wind_variability_hour, x='hour', y=['solar', 'wind'], title='Hourly Variability of Solar and Wind (Std Dev)', labels={'value': 'Standard Deviation', 'variable': 'Source'}), use_container_width=True)
    st.plotly_chart(px.bar(solar_wind_variability_year, x='year', y=['solar', 'wind'], title='Yearly Variability of Solar and Wind (Std Dev)', barmode='group'), use_container_width=True)

    source_columns = ['natural_gas', 'lignite', 'asphaltite_coal', 'hard_coal', 'fuel_oil', 'coal_imported', 'hydro_dam', 'hydro_river', 'solar', 'wind', 'biomass', 'geothermal', 'waste_heat', 'international']
    price_corr = df[source_columns + ['TRY/MWh']].corr()['TRY/MWh'].drop('TRY/MWh').sort_values(ascending=False).reset_index()
    price_corr.columns = ['Source', 'Correlation']
    st.subheader("💸 The Most Resource whic Associated with High Pricing Periods")
    st.plotly_chart(px.bar(price_corr, x='Source', y='Correlation', title='Correlation of Energy Sources with Electricity Price (TRY/MWh)', color='Correlation', color_continuous_scale='Reds'), use_container_width=True)

    demand_price_corr = df[['consumption_MWh', 'TRY/MWh']].corr().iloc[0, 1]
    st.subheader("💸 Correlation of Electricity Prices with Demand Levels")
    st.plotly_chart(px.scatter(df, x='consumption_MWh', y='TRY/MWh', title='Correlation Between Demand and Electricity Price', labels={'consumption_MWh': 'Demand (MWh)', 'TRY/MWh': 'Price (TRY/MWh)'}), use_container_width=True)

    seasonal_price = df.groupby('season')['TRY/MWh'].mean().reset_index()
    seasonal_price['Season'] = seasonal_price['season'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
    yearly_price = df.groupby('year')['TRY/MWh'].mean().reset_index()
    st.subheader("💸 Electricity Prices Trend through Seasonal and Annual")
    st.plotly_chart(px.bar(seasonal_price, x='Season', y='TRY/MWh', title='Average Electricity Price by Season', labels={'TRY/MWh': 'Price (TRY/MWh)'}), use_container_width=True)
    st.plotly_chart(px.line(yearly_price, x='year', y='TRY/MWh', title='Average Electricity Price by Year', labels={'TRY/MWh': 'Price (TRY/MWh)'}), use_container_width=True)

    features = ['consumption_MWh', 'total_renewable_enegry_MWh', 'total_nonrenewable_enegry_MWh'] + source_columns
    price_feature_corr = df[features + ['TRY/MWh']].corr()['TRY/MWh'].drop('TRY/MWh').sort_values(ascending=False).reset_index()
    price_feature_corr.columns = ['Feature', 'Correlation']
    st.subheader("💸 The Strongest Predictors of Electricity Price")
    st.plotly_chart(px.bar(price_feature_corr, x='Feature', y='Correlation', title='Correlation of Features with TRY/MWh Price', color='Correlation', color_continuous_scale='RdBu'), use_container_width=True)

    renew_vs_nonrenew_yearly = df.groupby('year')[['total_renewable_enegry_MWh', 'total_nonrenewable_enegry_MWh']].sum().reset_index()
    st.subheader("💸 Change of Resources Over Time")
    st.plotly_chart(px.line(renew_vs_nonrenew_yearly, x='year', y=['total_renewable_enegry_MWh', 'total_nonrenewable_enegry_MWh'], title='Total Electricity Generation: Renewable vs Nonrenewable (2018–2023)', labels={'value': 'Total Generation (MWh)', 'variable': 'Source'}), use_container_width=True)

    renewables = ['hydro_dam', 'hydro_river', 'solar', 'wind', 'biomass', 'geothermal', 'waste_heat', 'international']
    renewable_yearly_sources = df.groupby('year')[renewables].sum().reset_index()
    seasonal_renewable_sources = df.groupby('season')[renewables].sum().reset_index()
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    seasonal_renewable_sources['Season'] = seasonal_renewable_sources['season'].map(season_map)
    st.subheader("💸 The Most Renewable Source Contribute Per Year")
    st.plotly_chart(px.line(renewable_yearly_sources, x='year', y=renewables, title='Yearly Renewable Energy Contribution by Source'), use_container_width=True)
    st.plotly_chart(px.bar(seasonal_renewable_sources, x='Season', y=renewables, title='Seasonal Renewable Energy Contribution by Source', barmode='group'), use_container_width=True)

    melted_season = seasonal_renewable_sources.melt(id_vars='Season', var_name='Source', value_name='MWh')
    st.subheader("💸 Seasonal Generation Pattern for Renewable Resources")
    st.plotly_chart(px.bar(melted_season, x='Season', y='MWh', color='Source', title='Seasonal Generation Patterns of Renewable Sources', barmode='group'), use_container_width=True)

    renew_vs_demand = df.groupby(['year', 'season'])[['total_renewable_enegry_MWh', 'consumption_MWh']].sum().reset_index()
    renew_vs_demand['Surplus'] = renew_vs_demand['total_renewable_enegry_MWh'] - renew_vs_demand['consumption_MWh']
    renew_vs_demand['Season'] = renew_vs_demand['season'].map(season_map)
    st.subheader("💸 Showing If Renewable energy Can Meet T.Electricity Demand or Not")
    st.plotly_chart(px.bar(renew_vs_demand, x='Season', y='Surplus', color='year', title='Can Renewable Energy Alone Meet Seasonal Demand?'), use_container_width=True)

    seasonal_std = df.groupby('season')[['total_renewable_enegry_MWh', 'total_nonrenewable_enegry_MWh']].std().reset_index()
    seasonal_std['Season'] = seasonal_std['season'].map(season_map)
    st.subheader("💸 Comparing the variability of renewable generation to nonrenewable across seasons")
    st.plotly_chart(px.bar(seasonal_std, x='Season', y=['total_renewable_enegry_MWh', 'total_nonrenewable_enegry_MWh'], title='Seasonal Variability of Renewable vs Nonrenewable', barmode='group'), use_container_width=True)

def show_model_results(df, model, scaler):
    st.title("🎯 Model Performance")
    X = df[['hour', 'day_of_week', 'month', 'season', 'consumption_MWh', 'natural_gas', 'lignite', 'coal_imported', 'fuel_oil', 'wind', 'solar', 'hydro_dam', 'hydro_river', 'biomass', 'geothermal', 'total_renewable_enegry_MWh', 'total_nonrenewable_enegry_MWh', 'renewable_share', 'TRY/MWh']]
    y = df['renewable_strength_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    rf_pred = model.predict(X_test_scaled)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)


    lr_pred = model.predict(X_test_scaled)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    r2_lr = r2_score(y_test, y_pred_lr)
    rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)

    model_option = st.selectbox("Choose Model to View:", ["Random Forest", "Linear Regression", "Compare Both"])

    if model_option == "Random Forest":
        st.metric("R²", f"{r2_rf:.3f}")
        st.metric("RMSE", f"{rmse_rf:.2f}")

        st.subheader("📈 Actual vs Predicted (Random Forest)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred_rf, alpha=0.5, color='royalblue', edgecolors='k')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_title('Random Forest: Predicted vs Actual')
        ax.set_xlabel("Actual Renewable Strength Score")
        ax.set_ylabel("Predicted Renewable Strength Score")
        st.pyplot(fig)

    elif model_option == "Linear Regression":
        st.metric("R²", f"{r2_lr:.3f}")
        st.metric("RMSE", f"{rmse_lr:.2f}")

        st.subheader("📈 Actual vs Predicted (Linear Regression)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred_lr, alpha=0.5, color='green', edgecolor='gray')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual Renewable Strength Score")
        ax.set_ylabel("Predicted Renewable Strength Score")
        ax.set_title("Linear Regression: Predicted vs Actual")
        st.pyplot(fig)

    else:
        st.subheader("📊 Metrics Comparison")
        st.dataframe(pd.DataFrame({
            "Model": ["Random Forest", "Linear Regression"],
            "R²": [r2_rf, r2_lr],
            "RMSE": [rmse_rf, rmse_lr]
        }))

        st.subheader("📈 Actual vs Predicted – Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred_rf, alpha=0.4, label="Random Forest", color="royalblue", edgecolor='k')
        ax.scatter(y_test, y_pred_lr, alpha=0.4, label="Linear Regression", color="green", edgecolor='gray')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Perfect Fit")
        ax.legend()
        ax.set_xlabel("Actual Renewable Strength Score")
        ax.set_ylabel("Predicted Renewable Strength Score")
        st.pyplot(fig)


def show_feature_importance(df, model):
    st.title("🧠 Feature Importance")
    X = df[['hour', 'day_of_week', 'month', 'season', 'consumption_MWh', 'natural_gas', 'lignite', 'coal_imported', 'fuel_oil', 'wind', 'solar', 'hydro_dam', 'hydro_river', 'biomass', 'geothermal', 'total_renewable_enegry_MWh', 'total_nonrenewable_enegry_MWh', 'renewable_share', 'TRY/MWh']]

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = np.array(X.columns)[sorted_idx]
    sorted_importances = importances[sorted_idx]

    st.subheader("📈 Outlaires of Renewable Resources")
    fig, ax = plt.subplots(figsize=(8, 4))
    renewable_columns = ['hydro_dam', 'hydro_river', 'solar', 'wind', 'biomass', 'geothermal', 'waste_heat', 'international']
    sns.boxplot(data=df[renewable_columns])
    st.pyplot(fig)

    st.subheader("📈 Outlaires of Nonrenewable Resources")
    fig, ax = plt.subplots(figsize=(8, 4))
    nonrenewable_columns = ['natural_gas', 'lignite', 'asphaltite_coal', 'hard_coal', 'fuel_oil', 'coal_imported']
    sns.boxplot(data=df[nonrenewable_columns])
    st.pyplot(fig)

    st.subheader("🧱 Correlation Heatmap of Renewable Resourcse")
    fig, ax = plt.subplots(figsize=(10, 6))
    renewable_columns = ['hydro_dam', 'hydro_river', 'solar', 'wind', 'biomass', 'geothermal', 'waste_heat', 'international']
    renew_corr = df[renewable_columns].corr()
    sns.heatmap(renew_corr, annot=True, cmap='YlGnBu', fmt=".2f", square=True)
    st.pyplot(fig)

    st.subheader("🧱 Correlation Heatmap of Nonrenewable Resourcse")
    fig, ax = plt.subplots(figsize=(10, 6))
    nonrenewable_columns = ['natural_gas', 'lignite', 'asphaltite_coal', 'hard_coal', 'fuel_oil', 'coal_imported']
    nonrenew_corr = df[nonrenewable_columns].corr()
    sns.heatmap(nonrenew_corr, annot=True, cmap='OrRd', fmt=".2f", square=True)
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