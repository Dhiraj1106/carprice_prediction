from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and metadata from disk
with open("car_price_model.pkl", "rb") as f:
    saved = pickle.load(f)

model          = saved["model"]
ohe            = saved["ohe"]
companies      = saved["companies"]
fuel_types     = saved["fuel_types"]
company_models = saved.get("company_models", {})
r2             = saved["r2"]
mae            = saved["mae"]
rmse           = saved["rmse"]
train_size     = saved.get("train_size", 652)
year_min       = saved.get("year_min", 1995)
year_max       = saved.get("year_max", 2019)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/meta")
def meta():
    """Returns model stats, dropdown options, and company->model map."""
    return jsonify({
        "companies":      companies,
        "fuel_types":     fuel_types,
        "company_models": company_models,
        "r2":             r2,
        "mae":            mae,
        "rmse":           rmse,
        "train_size":     str(train_size),
        "year_min":       year_min,
        "year_max":       year_max,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Accepts car details and returns estimated resale price."""
    body = request.get_json()

    company = body["company"]
    year    = int(body["year"])
    kms     = int(body["kms_driven"])
    fuel    = body["fuel_type"]
    car_age = 2024 - year

    cat_features = ohe.transform([[company, fuel]])
    num_features = np.array([[year, kms, car_age]])
    X = np.hstack([cat_features, num_features])

    price = int(model.predict(X)[0])
    price = max(30000, price)

    return jsonify({
        "price":     price,
        "price_l":   round(price / 100000, 2),
        "low":       int(price * 0.8),
        "high":      int(price * 1.2),
        "car_age":   car_age,
        "dep_pct":   round(car_age * 4.2, 1),
        "fuel_mult": 1.0
    })


@app.route("/api/retrain", methods=["POST"])
def retrain():
    """Retrain model on demand."""
    try:
        import subprocess, sys
        result = subprocess.run([sys.executable, "car_price_prediction.py"],
                                capture_output=True, text=True, timeout=60)
        # Reload
        global model, ohe, companies, fuel_types, company_models, r2, mae, rmse, train_size
        with open("car_price_model.pkl", "rb") as f:
            saved = pickle.load(f)
        model          = saved["model"]
        ohe            = saved["ohe"]
        companies      = saved["companies"]
        fuel_types     = saved["fuel_types"]
        company_models = saved.get("company_models", {})
        r2             = saved["r2"]
        mae            = saved["mae"]
        rmse           = saved["rmse"]
        train_size     = saved.get("train_size", 652)
        return jsonify({"r2": r2, "mae": mae, "rmse": rmse})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
