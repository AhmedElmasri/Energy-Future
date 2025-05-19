🔌 Powering Sustainability: Forecasting Electricity Prices and Renewable Energy Trends

📈 Overview

This project uses machine learning to forecast electricity prices and analyze renewable energy trends in Turkey from 2018 to 2023.  
We leverage real-world electricity generation and consumption data to build accurate regression models and uncover sustainability insights.

🎯 Objectives
- Predict hourly electricity prices (TRY/MWh) using regression models
- Analyze contributions of renewable vs fossil sources
- Explore consumption patterns (hourly, monthly, seasonal)
- Support sustainability insights with real data

🧠 Machine Learning Models
- Linear Regression (Baseline)
- Random Forest Regressor (Best performance)

Evaluation Metrics:
- R² Score
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

🛠️ Techniques Used
- Data cleaning & outlier handling
- Time-based feature engineering (hour, day, month, season)
- StandardScaler for feature normalization
- Model training, evaluation, and visualization

📊 Key Visualizations
- Hourly & Monthly Electricity Demand
- Fossil vs Renewable Generation (Pie Chart)
- Solar/Wind Generation by Hour
- Annual Electricity Prices
- Actual vs Predicted Prices (LR & RF)
- Feature Importance (Random Forest)

💻 Tools & Libraries
- Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly
- scikit-learn (for ML)
- Google Colab

📦 Dataset
Contains hourly data of electricity generation by source, consumption, prices (TRY, USD, EUR), and more from 2018–2023 in Turkey.

🧩 Results
Model	                       R² Score	        MAE (TRY       RMSE (TRY)
Linear Regression	            0.793	           124.58	         172.93
Random Forest             	  0.976            27.34	         59.47

📌 Conclusion
Machine learning offers powerful insights into electricity pricing and energy mix optimization.  
Random Forest achieved the highest accuracy, and renewables showed strong growth and price stability.
