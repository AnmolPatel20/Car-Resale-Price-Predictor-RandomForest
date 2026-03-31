# 🚗 Pre-Owned Vehicle Price Estimation

![Car Price GIF](images/CarPriceProject.gif)

> Predict the resale price of a used car based on its features — helping sellers make smarter pricing decisions based on real market data.

---

## 📌 Problem Statement

With thousands of used cars listed on platforms like **CarDekho**, it's hard for sellers to know the right price. This project builds a machine learning model that predicts the **selling price of a pre-owned vehicle** based on key features like engine size, mileage, fuel type, and more.

---

## 📂 Dataset

- **Source:** Scraped from [cardekho.com](https://www.cardekho.com)
- **Size:** 15,411 rows × 13 columns
- **Target Variable:** `selling_price`

| Feature | Description |
|---|---|
| `model` | Car model name |
| `vehicle_age` | Age of the vehicle (in years) |
| `km_driven` | Total kilometers driven |
| `seller_type` | Individual / Dealer / Trustmark Dealer |
| `fuel_type` | Petrol / Diesel / CNG / LPG |
| `transmission_type` | Manual / Automatic |
| `mileage` | Fuel efficiency (km/l) |
| `engine` | Engine displacement (cc) |
| `max_power` | Maximum power (bhp) |
| `seats` | Number of seats |

---

## 📊 Exploratory Data Analysis

### Selling Price Distribution
![Selling Price Distribution](images/selling_price_distribution.png)

Most cars are priced between ₹2–15 lakhs, with the distribution heavily right-skewed indicating a few high-end luxury cars in the dataset.

### Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)

`max_power` (0.75) and `engine` (0.59) are the strongest predictors of selling price. `vehicle_age` and `km_driven` show a negative correlation as expected.

---

## ⚙️ Preprocessing

- Dropped `car_name` and `brand` columns (redundant)
- Applied **Label Encoding** on `model` column (120 unique values)
- Applied **One-Hot Encoding** on `seller_type`, `fuel_type`, `transmission_type`
- Applied **Standard Scaling** on all numerical features
- Used `ColumnTransformer` pipeline for clean and consistent transformations

---

## 🤖 Model Training & Comparison

Trained and evaluated **6 regression models**:

![Model Comparison](images/model_comparison.png)

| Model | Train R² | Test R² |
|---|---|---|
| Linear Regression | 0.6218 | 0.6645 |
| Lasso | 0.6218 | 0.6645 |
| Ridge | 0.6218 | 0.6645 |
| K-Neighbors Regressor | 0.8691 | 0.9150 |
| Decision Tree | 0.9995 | 0.8823 |
| **Random Forest** | **0.9791** | **0.9303** |

✅ **Random Forest Regressor** was selected as the best model — high test R², no overfitting.

---

## 🔧 Hyperparameter Tuning

Used `RandomizedSearchCV` on KNN and Random Forest with 3-fold cross validation.

**Best params for Random Forest:**
```
n_estimators: 1000
min_samples_split: 2
max_features: 7
max_depth: None
```

**Final Model Performance after tuning:**
| Metric | Train | Test |
|---|---|---|
| R² Score | 0.9804 | 0.9403 |
| RMSE | ₹1,26,209 | ₹2,12,015 |
| MAE | ₹38,890 | ₹98,050 |

---

## 💾 Model Saving

```python
import joblib

joblib.dump(rf_model, 'car_price_predictor.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
```

---

## 🔮 Sample Prediction

```python
import joblib
import pandas as pd

rf_model = joblib.load('car_price_predictor.pkl')
preprocessor = joblib.load('preprocessor.pkl')

sample = pd.DataFrame([{
    'model': 38,
    'vehicle_age': 6,
    'km_driven': 30000,
    'seller_type': 'Dealer',
    'fuel_type': 'Diesel',
    'transmission_type': 'Manual',
    'mileage': 22.77,
    'engine': 1498,
    'max_power': 98.59,
    'seats': 5
}])

sample_transformed = preprocessor.transform(sample)
predicted_price = rf_model.predict(sample_transformed)
print(f"Predicted Selling Price: ₹ {predicted_price[0]:,.0f}")
# Output: Predicted Selling Price: ₹ 6,08,500
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)
![Seaborn](https://img.shields.io/badge/Seaborn-Viz-lightblue)

---

## 📁 Project Structure

```
├── images/
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   └── selling_price_distribution.png
├── .gitignore
├── Resale_Car_Prediction.ipynb
├── car_price_predictor.pkl
├── cardekho_imputated.csv
├── preprocessor.pkl
└── README.md
```
