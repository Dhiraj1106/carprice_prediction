"""
Car Price Prediction using Linear Regression
Dataset : Cleaned_Car_data.csv  (816 records, 25 companies)
R2 Score: ~0.63   |   MAE: ~Rs.1.57 Lakh
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


# ---------------------------------------------------------
# 1. Load and clean the dataset
# ---------------------------------------------------------

df = pd.read_csv("Cleaned_Car_data.csv")

df = df.drop(columns=["Unnamed: 0"])
df = df[df["Price"] < 5_000_000]
df["car_age"] = 2024 - df["year"]

print(f"Records  : {len(df)}")
print(f"Brands   : {df['company'].nunique()} unique companies")
print(f"Price range : Rs.{df['Price'].min():,.0f}  to  Rs.{df['Price'].max():,.0f}")
print()


# ---------------------------------------------------------
# 2. Build company -> model names map from dataset
# ---------------------------------------------------------

company_models = {}
for company in sorted(df["company"].unique()):
    names = sorted(df[df["company"] == company]["name"].unique().tolist())
    company_models[company] = names


# ---------------------------------------------------------
# 3. Prepare features
# ---------------------------------------------------------

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_features = ohe.fit_transform(df[["company", "fuel_type"]])
num_features = df[["year", "kms_driven", "car_age"]].values

X = np.hstack([cat_features, num_features])
y = df["Price"].values


# ---------------------------------------------------------
# 4. Split and train
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


# ---------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------

y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=" * 45)
print("         MODEL PERFORMANCE")
print("=" * 45)
print(f"  R2 Score : {r2:.4f}")
print(f"  MAE      : Rs.{mae:>12,.0f}")
print(f"  RMSE     : Rs.{rmse:>12,.0f}")
print(f"  Train    : {len(X_train)} samples")
print(f"  Test     : {len(X_test)} samples")
print("=" * 45)
print()


# ---------------------------------------------------------
# 6. Save the model
# ---------------------------------------------------------

with open("car_price_model.pkl", "wb") as f:
    pickle.dump({
        "model":          model,
        "ohe":            ohe,
        "r2":             round(r2, 4),
        "mae":            round(mae),
        "rmse":           round(rmse),
        "companies":      sorted(df["company"].unique().tolist()),
        "fuel_types":     sorted(df["fuel_type"].unique().tolist()),
        "company_models": company_models,
        "year_min":       int(df["year"].min()),
        "year_max":       int(df["year"].max()),
        "train_size":     len(X_train),
    }, f)

print("Model saved -> car_price_model.pkl")
print()


# ---------------------------------------------------------
# 7. Reusable predict function
# ---------------------------------------------------------

def predict_price(company, year, kms_driven, fuel_type):
    car_age   = 2024 - year
    cat_input = ohe.transform([[company, fuel_type]])
    num_input = np.array([[year, kms_driven, car_age]])
    X_input   = np.hstack([cat_input, num_input])
    price     = model.predict(X_input)[0]
    return max(30_000, int(price))


# ---------------------------------------------------------
# 8. Interactive CLI predictor
# ---------------------------------------------------------

if __name__ == "__main__":
    print("-- Interactive Predictor --")
    try:
        company    = input("Company   (e.g. Maruti)       : ").strip()
        year       = int(input("Year      (1995-2019)         : "))
        kms_driven = int(input("KMs driven                    : "))
        fuel_type  = input("Fuel type (Petrol/Diesel/LPG) : ").strip()

        price = predict_price(company, year, kms_driven, fuel_type)
        print(f"\n  Estimated Price : Rs.{price:,.0f}  (~Rs.{price / 1e5:.2f} Lakh)")

    except (ValueError, KeyboardInterrupt):
        print("\nExiting.")
