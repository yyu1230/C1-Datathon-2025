
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Load and preprocess data ==========
traffic_df = pd.read_csv("monthly_transportation_statistics.csv")
gasoline_df = pd.read_csv("weekly_gasoline_prices.csv", usecols=["Date", "Price"], parse_dates=["Date"])

traffic_df['Date'] = pd.to_datetime(traffic_df['Date'], errors='coerce')
traffic_df['Quarter'] = traffic_df['Date'].dt.to_period('Q')
quarter_df = traffic_df[['Quarter', 'Highway Fatalities', 
                         'Unemployment Rate - Seasonally Adjusted',
                         'Auto sales SAAR (millions)']].dropna()
quarter_df = quarter_df.groupby('Quarter').agg({
    'Highway Fatalities': 'mean',
    'Unemployment Rate - Seasonally Adjusted': 'mean',
    'Auto sales SAAR (millions)': 'mean'
}).reset_index()

gasoline_df['Quarter'] = gasoline_df['Date'].dt.to_period('Q')
quarterly_gas = gasoline_df.groupby('Quarter')['Price'].mean().reset_index()

df_q = quarter_df.merge(quarterly_gas, on='Quarter')
df_q['Quarter'] = df_q['Quarter'].dt.to_timestamp()
df_q = df_q.rename(columns={
    'Quarter': 'Date',
    'Highway Fatalities': 'Crashes',
    'Unemployment Rate - Seasonally Adjusted': 'Unemployment',
    'Auto sales SAAR (millions)': 'AutoSales'
})

for lag in range(1, 5):
    df_q[f'Price_Lag_{lag}'] = df_q['Price'].shift(lag)

df_q['TimeTrend'] = (df_q['Date'] - df_q['Date'].min()).dt.days
df_q['QuarterNum'] = df_q['Date'].dt.quarter
quarter_dummies = pd.get_dummies(df_q['QuarterNum'], prefix='Q', drop_first=True)
feature_df_q = pd.concat([df_q, quarter_dummies], axis=1).dropna().reset_index(drop=True)

# ========== Modeling ==========
X_cols = [f'Price_Lag_{lag}' for lag in range(1, 5)] + ['Unemployment', 'AutoSales', 'TimeTrend'] + list(quarter_dummies.columns)
X = feature_df_q[X_cols]
y = feature_df_q['Crashes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

results = []

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': mean_squared_error(y_test, y_pred, squared=False),
        'R2': r2_score(y_test, y_pred)
    })

# SVR (special case: scale first)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_svr, X_test_svr = X_scaled[:len(X_train)], X_scaled[len(X_train):]
svr = SVR(kernel='rbf', C=100, gamma=0.1)
svr.fit(X_train_svr, y_train)
svr_pred = svr.predict(X_test_svr)
results.append({
    'Model': 'SVR',
    'MAE': mean_absolute_error(y_test, svr_pred),
    'RMSE': mean_squared_error(y_test, svr_pred, squared=False),
    'R2': r2_score(y_test, svr_pred)
})

# ========== Plot and export results ==========
results_df = pd.DataFrame(results).sort_values(by='MAE')
plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='MAE', data=results_df)
plt.title('Quarterly Prediction - MAE of Different Models')
plt.tight_layout()
plt.savefig("model_comparison_plot.png")
results_df.to_csv("model_comparison_results.csv", index=False)
