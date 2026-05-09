import pandas as pd
import numpy as np
import time
import tracemalloc
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load and filter to Virginia
print("Loading PM2.5 data...")
pm = pd.read_csv('hourly_88101_2025.csv')
pm = pm[pm['State Name'] == 'Virginia']

print("Loading Ozone data...")
oz = pd.read_csv('hourly_44201_2025.csv')
oz = oz[oz['State Name'] == 'Virginia']

# Keep only what is needed
pm = pm[['Date Local','Time Local','Latitude','Longitude','Sample Measurement']].rename(columns={'Sample Measurement':'pm25'})
oz = oz[['Date Local','Time Local','Latitude','Longitude','Sample Measurement']].rename(columns={'Sample Measurement':'o3'})

# Merge on date/time/location
df = pd.merge(pm, oz, on=['Date Local','Time Local','Latitude','Longitude'])
df['datetime'] = pd.to_datetime(df['Date Local'] + ' ' + df['Time Local'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"Rows after merge: {len(df)}")

# Feature engineering 
df['hour']        = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month']       = df['datetime'].dt.month
df['pm25_lag1']   = df['pm25'].shift(1)
df['o3_lag1']     = df['o3'].shift(1)
df = df.dropna()

print(f"Rows after feature engineering: {len(df)}")

features = ['hour','day_of_week','month','Latitude','Longitude','pm25_lag1','o3_lag1']
X = df[features].values
y = df['pm25'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
    'XGBoost':           xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    tracemalloc.start()
    _ = model.predict(X_test)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_mb = peak / 1024 / 1024

    t0 = time.perf_counter()
    _ = model.predict(X_test)
    elapsed = time.perf_counter() - t0
    ms_per_sample = (elapsed / len(X_test)) * 1000

    pickle.dump(model, open('/tmp/tmp_model.pkl','wb'))
    size_mb = os.path.getsize('/tmp/tmp_model.pkl') / 1024 / 1024

    results[name] = {
        'RMSE': round(rmse,3), 'MAE': round(mae,3), 'R2': round(r2,4),
        'Memory_MB': round(mem_mb,3), 'Inference_ms': round(ms_per_sample,4),
        'Size_MB': round(size_mb,3), 'y_pred': y_pred
    }
    print(f"  RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}  Mem={mem_mb:.2f}MB  Inf={ms_per_sample:.4f}ms  Size={size_mb:.3f}MB")

# Results table 
rows = [{'Model':n,'RMSE':r['RMSE'],'MAE':r['MAE'],'R2':r['R2'],
         'Memory_MB':r['Memory_MB'],'Inference_ms':r['Inference_ms'],'Size_MB':r['Size_MB']}
        for n,r in results.items()]
pd.DataFrame(rows).to_csv('results_table.csv', index=False)
print("\nSaved results_table.csv")

# Plot 
palette = ['#1a6faf','#c0392b','#27ae60','#e67e22']
model_names = list(results.keys())
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Edge-Based Air Quality Model Comparison (Virginia PM2.5)', fontweight='bold')

ax = axes[0,0]
vals = [results[m]['RMSE'] for m in model_names]
bars = ax.bar(model_names, vals, color=palette)
ax.set_title('RMSE (lower = better)'); ax.set_ylabel('RMSE')
ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=8)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.3f}', ha='center', fontsize=8)

ax = axes[0,1]
vals = [results[m]['R2'] for m in model_names]
bars = ax.bar(model_names, vals, color=palette)
ax.set_title('R2 Score (higher = better)'); ax.set_ylabel('R2'); ax.set_ylim(0,1.1)
ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=8)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.3f}', ha='center', fontsize=8)

ax = axes[1,0]
vals = [results[m]['Inference_ms'] for m in model_names]
bars = ax.bar(model_names, vals, color=palette)
ax.set_title('Inference Time (ms/sample)'); ax.set_ylabel('ms')
ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=8)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02, f'{v:.4f}', ha='center', fontsize=8)

ax = axes[1,1]
vals = [results[m]['Size_MB'] for m in model_names]
bars = ax.bar(model_names, vals, color=palette)
ax.set_title('Model Size (MB)'); ax.set_ylabel('MB')
ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=8)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.2f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('results_figure.png', dpi=150, bbox_inches='tight')
print("Saved results_figure.png")
