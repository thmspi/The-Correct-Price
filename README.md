# The Correct Price — Car Price Prediction

A machine learning project that predicts used car selling prices using regression and K-Nearest Neighbors models trained on a dataset of ~550 000 auction records.

## Dataset

**`car_prices.csv`** — Used car auction sales data with the following columns:

| Column | Description |
|---|---|
| `year` | Model year of the vehicle |
| `make` | Manufacturer (e.g. BMW, Kia, Audi) |
| `model` | Model name |
| `trim` | Trim level |
| `body` | Body style (Sedan, SUV, Coupe, …) |
| `transmission` | Transmission type (automatic / manual) |
| `vin` | Vehicle Identification Number |
| `state` | US state of the auction |
| `condition` | Vehicle condition score (1–49) |
| `odometer` | Mileage |
| `color` | Exterior colour |
| `interior` | Interior colour |
| `seller` | Seller name |
| `mmr` | Manheim Market Report — wholesale price estimate |
| `sellingprice` | **Target variable** — actual auction selling price |
| `saledate` | Date and time of the auction |

## Notebooks

### `Regression_car_10K.ipynb` & `Regression_car_100K.ipynb`
Linear and Polynomial regression pipelines run on 10 000 and 100 000 rows respectively.

**Pipeline steps:**
1. Drop irrelevant columns (`vin`, `trim`, `saledate`, `state`, `mmr`)
2. Impute missing values (mean for numeric, mode for categorical)
3. Encode categorical features with `pd.factorize`
4. Filter odometer outliers (> 500 000 km) and apply log-transform
5. Normalise with `MinMaxScaler`
6. Fit **Linear Regression** and evaluate with R², MAE, MSE, RMSE
7. Fit **Polynomial Regression** (degree = 6) and compare metrics

### `KNN_car.ipynb`
K-Nearest Neighbors regression pipeline on 100 000 rows.

**Pipeline steps:**
1–5. Same preprocessing as above (categorical columns `make`, `body`, `model`, `transmission`, `color` are dropped after correlation analysis)
6. Fit a baseline **KNN Regressor** (`sklearn` defaults)
7. Tune hyperparameters with `RandomizedSearchCV` over:
   - `n_neighbors` (1–29)
   - `weights` (uniform / distance)
   - `algorithm` (auto, ball_tree, kd_tree, brute)
   - `leaf_size` (30–49)
   - `p` (1 = Manhattan, 2 = Euclidean)

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

## Usage

Open any notebook with Jupyter and run all cells. The dataset `car_prices.csv` must be in the same directory.

```bash
jupyter notebook Regression_car_100K.ipynb
```

## Project Report

A full written analysis of the methodology and results is available in `Rapport_ML_vf.docx`.
