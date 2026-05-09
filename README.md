# Edge-Based Air Quality Prediction Under Resource Constraints

Evaluates four lightweight ML models (Linear Regression, Random Forest, Gradient
Boosting, XGBoost) for PM2.5 forecasting under simulated edge computing constraints,
comparing prediction accuracy, inference speed, memory usage, and model size.

## Data

Download from the EPA AQS website (https://aqs.epa.gov/aqsweb/airdata/download_files.html)
and place in the same directory as `main.py`:
- `hourly_88101_2025.csv` (PM2.5)
- `hourly_44201_2025.csv` (Ozone)

## Usage

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
python main.py
```

Outputs `results_table.csv`, `results_figure.png`, and `results_figure2.png`.

## Author
Allyson Morris
ammorris02@wm.edu
